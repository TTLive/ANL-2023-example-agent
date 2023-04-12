import decimal
import logging
import math
from random import randint, choice
from time import time
from typing import cast
import numpy as np

import numpy
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
        self.good_bids_values = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.previous_bids: list[Bid] = []
        self.last_received_bid: Bid = None
        self.last_offered_bid: Bid = None
        self.mirrored_vector: list[int] = None
        self.count_since_paret = 3
        self.paretoBids = None

        self.count_bid = 0

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


            # gets all good bids with a utility threshold of 0.45
            self.all_good_bids = self.getAllGoodBids(AllBidsList(self.domain), 0.5)
            #print(f"\n the amount of good bids is: {len(self.all_good_bids)}\n")
            self.mirrored_vector = [0,0]
            # stores all utility values of bids in good_bids_values
            self.good_bids_values = np.array([x[1] for x in self.all_good_bids])
            # stores all bids in all_good_bids
            self.all_good_bids = [x[0] for x in self.all_good_bids]
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
            
            # Get a mirrored vector from the opponents bid relative to previous bid
            # Checks if opponent has a previous bid, otherwise not using a mirrored bid
            if self.last_received_bid:
                # gets the estimated utility value of the opponent from their bid
                opponent_val = self.opponent_model.get_predicted_utility(bid) - self.opponent_model.get_predicted_utility(self.last_received_bid)


                # gets the agents utility value from their bid
                received_val = self.profile.getUtility(bid) - self.profile.getUtility(self.last_received_bid)
                # gets a mirrored vector as a mirrored bid
                self.mirrored_vector = self.get_mirrored_vector(received_val, decimal.Decimal(str(opponent_val)))
            self.last_received_bid = bid
            # set bid as last received
           

    def get_mirrored_vector(self, our_val, opp_val):
        progress = self.progress.get(time() * 1000)

        # depending on progress and direction of their bid, changes the length
        # lower utility value for our agent means we make a bid that has a lower utility value for the opponent
        # as progress increases then vector becomes shorter and vice versa
        if our_val < 0:
            progress = 1.5 - progress
        else:
            progress = 0.5 + progress

        length = math.sqrt(opp_val * opp_val + our_val * our_val)
        our = 0
        opp = 0
        if our_val != 0:
            our = float(our_val) * progress
        if opp_val != 0:
            opp = float(opp_val) * progress
        # returns opp, val to get the mirrored vector
        return our, opp

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        bid_util = 0
        if self.last_received_bid:
            bid_util = self.profile.getUtility(self.last_received_bid)

        # check if the last received offer is good enough according to simple conditions
        if self.accept_condition(bid_util):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            #find a bid
            my_bid = self.find_bid()
            # accept if opponents last bid is better than your next bid
            if bid_util >= self.profile.getUtility(my_bid):
                action = Accept(self.me, self.last_received_bid)
            #otherwise offer our bid
            else:
                self.previous_bids.append(my_bid)
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

    def accept_condition(self, bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)
        bid_util = bid
        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            # threshold for accepting decreases over time [0.9 to 0.65]
            bid_util > (0.9 - (progress / 4)),

            # accept if deadline nearly ended
            # Maybe change this to calculate avg turn time from previous turns and then if there is not enough time left accept
            progress > 0.9 and bid_util > 0.4,
        ]
        return any(conditions)

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        progress = self.progress.get(time() * 1000)
        #first turn -> make best possbile bid
        if len(self.previous_bids) < 1:
            best_bid = self.all_good_bids[-1]
            self.good_bids_oppValues = np.array([self.opponent_model.get_predicted_utility(x) for x in self.all_good_bids])
            return best_bid
        #Next couple of turns make random bids with high utility
        elif progress < 0.02:
            start = np.argmax(self.good_bids_values > 0.8)
            rand_bid = choice(self.all_good_bids[start:])
            if rand_bid:
                return rand_bid
            else:
                return self.previous_bids[-1]
        elif progress < 0.9:
            #slow linear progress
            # only consider bids from the thresholdvalue and forward, threshold slowly decreases over time
            threshold = 0.7 - (progress * 0.15)
            #each turn calculate opponent bid util since it might have changed when updating the model
            self.good_bids_oppValues = np.array([self.opponent_model.get_predicted_utility(x) for x in self.all_good_bids])
            inds = []
            #get opponent utility of mirrored vector placed upon our previous bid
            previous_bid_x = decimal.Decimal(str(self.opponent_model.get_predicted_utility(self.previous_bids[-1]))) + decimal.Decimal(str(self.mirrored_vector[0])) - decimal.Decimal(str(0.1))
            #find all indices in all_good_bids that satisfy both the threshold and the opponent utility value
            for i in range(len(self.all_good_bids)):
                if(self.good_bids_values[i] > threshold and self.good_bids_oppValues[i] > previous_bid_x):
                    inds.append(i)
            self.paretoBids = self.getEstimatedPareto(inds)
            closest_bid = self._closestPoint(self.previous_bids[-1], self.paretoBids, self.mirrored_vector)
            return closest_bid
        #faster progress
        #Do the same as previously but here the threshold will rapidly decrease, note that when below 0.45 it doesnt matter since the bids are not in all_good_bids
        else:
            threshold = 0.58 * (1.8 - progress)**2
            previous_bid_x = decimal.Decimal(str(self.opponent_model.get_predicted_utility(self.previous_bids[-1]))) + decimal.Decimal(str(self.mirrored_vector[0])) - decimal.Decimal(str(0.05))
            inds = []
            for i in range(len(self.all_good_bids)):
                if(self.good_bids_values[i] > threshold > self.mirrored_vector[0] and self.good_bids_oppValues[i] > previous_bid_x):
                    inds.append(i)
            self.paretoBids = self.getEstimatedPareto(inds)
            closest_bid = self._closestPoint(self.previous_bids[-1], self.paretoBids, self.mirrored_vector)
            return closest_bid

    def score_bid(self, bid: Bid, alpha: float = 0.7, eps: float = 0.1) -> float:
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

    #returns all bids that have own utility above the threshold
    def getAllGoodBids(self, all_bids, threshold):
        bids : list(Bid, float) = []
        for bid in all_bids:
            util = self.profile.getUtility(bid)
            if util > threshold:
                bids.append((bid, util))
        bids.sort(key=lambda a: a[1])
        return bids

    #Using the estimated opponent model and own utility model get list of pareto bids from a list of bids
    #Note that this is inspired by the getPareto in create_domains.py
    def getEstimatedPareto(self, inds):
        pareto_front = []
        #save all non Pareto bids to find second rank later
        rank2 = []
        while inds:
            cand_ind = inds.pop(0)
            candidate_bid = self.all_good_bids[cand_ind]
            cand_bid_vals = [self.good_bids_values[cand_ind], self.good_bids_oppValues[cand_ind]]
            bid_nr = 0
            dominated = False
            while len(inds) != 0 and bid_nr < len(inds):
                bid_ind = inds[bid_nr]
                bid = self.all_good_bids[bid_ind]
                bid_vals = [self.good_bids_values[bid_ind], self.good_bids_oppValues[bid_ind]]
                if self._dominates(bid_vals, cand_bid_vals):
                    # If it is dominated remove the bid from all bids
                    inds.pop(bid_nr)
                    rank2.append(bid_ind)
                elif self._dominates(cand_bid_vals, bid_vals):
                    dominated = True
                    bid_nr += 1
                else:
                    bid_nr += 1

            if not dominated:
                # add the non-dominated bid to the Pareto frontier
                
                pareto_front.append(
                    {
                        "bid": candidate_bid,
                        "utility": [
                            self.good_bids_values[cand_ind], 
                            self.good_bids_oppValues[cand_ind],
                        ],
                    }
                )
            else:
                rank2.append(cand_ind)

        #Also find all second rank pareto bids by running pareto without the actual Pareto bids
        while rank2:
            cand_ind = rank2.pop(0)
            candidate_bid = self.all_good_bids[cand_ind]
            cand_bid_vals = [self.good_bids_values[cand_ind], self.good_bids_oppValues[cand_ind]]
            bid_nr = 0
            dominated = False
            while len(rank2) != 0 and bid_nr < len(rank2):
                bid_ind = rank2[bid_nr]
                bid = self.all_good_bids[bid_ind]
                bid_vals = [self.good_bids_values[bid_ind], self.good_bids_oppValues[bid_ind]]
                if self._dominates(bid_vals, cand_bid_vals):
                    rank2.pop(bid_nr)
                elif self._dominates(cand_bid_vals, bid_vals):
                    dominated = True
                    bid_nr += 1
                else:
                    bid_nr += 1

            if not dominated:
                
                pareto_front.append(
                    {
                        "bid": candidate_bid,
                        "utility": [
                            self.good_bids_values[cand_ind], 
                            self.good_bids_oppValues[cand_ind],
                        ],
                    }
                )
        #Sort descending on our utility
        pareto_front = reversed(sorted(pareto_front, key=lambda a: a["utility"][0]))

        return pareto_front

    # check if candidate_bid dominates bid
    def _dominates(self, bid, candidate_bid):
        if bid[0] > candidate_bid[0]:
            return False
        elif bid[1] > candidate_bid[1]:
            return False
        else:
            return True

    # finds the bid with the closest x value on the pareto frontier from the previous bid and vector
    def _closestPoint(self, bid, paretoFrontier, vector, step=[0, 0]):

        #finds the opponent utility of our mirrored vector (based on previous opponent bid) placed on our previous bid
        bid_opp_util = decimal.Decimal(str(self.opponent_model.get_predicted_utility(bid))) + decimal.Decimal(str(vector[0]))

        newBid = None
        for b in paretoFrontier:
            #should take the first bid that has same or better opponent utility as our mirrored vector
            if (bid_opp_util <= b["utility"][1]):
                newBid = b
                break
        if (newBid == None):
            return bid
        else:
            return newBid["bid"]

    def _nashProduct(self, paretoFrontier):
        ratios = []
        for b in paretoFrontier:
            ratios.append(numpy.abs(b["utility"][0] - b["utility"][1]))
        nash_index = ratios.index(numpy.minimum(ratios))
        return paretoFrontier[nash_index]["bid"]
