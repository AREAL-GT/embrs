

from embrs.utilities.fire_util import CrownStatus

# TODO: Fire sim should have an instance of Embers -> add to fire sim constructor
class Embers:
    def __init__(self, ign_prob: float):

        # Spotting Ignition Probability
        self.ign_prob = ign_prob # TODO: this needs to be a sim parameter
        self.spot_source = CrownStatus.NONE

        self.embers = []

    def loft(self):
        # Attempts to loft 16 embers of different types

        # Appends to self.embers

        pass

    def flight(self):
        # Loops through embers

            # Computes the landing and ignition decision for each

            # Determines whether an ember should carry-over to next iteration

        pass




    

