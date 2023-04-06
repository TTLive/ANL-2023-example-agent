import decimal
import logging
import math
from random import randint
from time import time
from typing import cast

import geniusweb
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel

class TemplateAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.all_good_bids: list(Bid, float) = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.last_offered_bid: Bid = None
        self.mirrored_vector: tuple[float, float] = None

        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            self.opponent_model = OpponentModel(self.domain)
            ### gets all good bids with a utility threshold of 0.7
            self.all_good_bids = self.getAllGoodBids(AllBidsList(self.domain), 0.7)
            pareto = self.getEstimatedPareto([x[0] for x in self.all_good_bids])

            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)

            if self.last_received_bid:
                test = self.opponent_model.get_predicted_utility(self.last_received_bid)
                test2 = self.opponent_model.get_predicted_utility(bid)
                received_val = self.opponent_model.get_predicted_utility(self.last_received_bid) \
                              - self.opponent_model.get_predicted_utility(bid)
                offered_val = self.profile.getUtility(self.last_received_bid) - self.profile.getUtility(bid)
                self.mirrored_vector = self.normalize(offered_val, decimal.Decimal(str(received_val)))

            # set bid as last received
            self.last_received_bid = bid

    def normalize(self, offered, received):
        length = math.sqrt(received*received + offered*offered)
        off = 0
        rec = 0
        if offered != 0:
            off = float(offered) / length
        if received != 0:
            rec = float(received) / length
        return off, rec

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        my_bid = self.find_bid()
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid, my_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            action = Offer(self.me, my_bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid, my_bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            # threshold for accepting decreases over time [0.9 to 0.65]
            self.profile.getUtility(bid) > (0.9 - (progress / 4)),
            # accept if deadline nearly ended
            progress > 0.95,
            # accept if opponents last bid is better than your next bid
            self.profile.getUtility(bid) > self.profile.getUtility(my_bid),
        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        all_bids = [x[0] for x in self.all_good_bids]

        best_bid_score = 0.0
        best_bid = None

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids[randint(0, len(all_bids) - 1)]
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        self.last_offered_bid = best_bid

        return best_bid

    def find_mirrored_bid(self) -> Bid:
        opponent_val = OpponentModel.get_predicted_utility(self.last_received_bid)
        my_val = self.profile.getUtility(self.last_received_bid)


    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score

    #todo optimise speed, could go in report
    def getAllGoodBids(self, all_bids, threshold):
        bids : list(Bid, float) = []
        for bid in all_bids:
            util = self.profile.getUtility(bid)
            if util > threshold:
                bids.append((bid, util))
        bids.sort(key=lambda a: a[1])
        return bids

    def getEstimatedPareto(self, bids):
        pareto_front = []
        # dominated_bids = set()
        while bids:
            candidate_bid = bids.pop(0)
            bid_nr = 0
            dominated = False
            while len(bids) != 0 and bid_nr < len(bids):
                bid = bids[bid_nr]

                if self._dominates(candidate_bid, bid):
                    # If it is dominated remove the bid from all bids
                    bids.pop(bid_nr)
                    # dominated_bids.add(frozenset(bid.items()))
                elif self._dominates(candidate_bid, bid):
                    dominated = True
                    # dominated_bids.add(frozenset(candidate_bid.items()))
                    bid_nr += 1
                else:
                    bid_nr += 1

            if not dominated:
                # add the non-dominated bid to the Pareto frontier
                pareto_front.append(
                    {
                        "bid": candidate_bid,
                        "utility": [
                            self.profile.getUtility(candidate_bid),
                            self.opponent_model.get_predicted_utility(candidate_bid),
                        ],
                    }
                )

        pareto_front = sorted(pareto_front, key=lambda d: d["utility"][0])

        return pareto_front

    def _dominates(self, bid, candidate_bid):
        if self.profile.getUtility(bid) < self.profile.getUtility(candidate_bid):
            return False
        elif self.opponent_model.get_predicted_utility(bid) < self.opponent_model.get_predicted_utility(
            candidate_bid
        ):
            return False
        else:
            return True

