class NoduleFinding(object):
    """
    Represents a nodule
    """

    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
                 CADprobability=None, noduleType=None, diameter=None, state=None, seriesInstanceUID=None):

        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.diameter_mm = diameter
        self.state = state
        self.candidateID = None
        self.seriesuid = seriesInstanceUID

    def list_all_member(self):
        """
        list all class member
        """
        print(60 * "*")
        for name, value in vars(self).items():
            print("{0} = {1}".format(name, value))

    def transform_inner_data_type(self):
        """
        transform inner datas from str to corresponding types
        """
        self.coordX = float(self.coordX)
        self.coordY = float(self.coordY)
        self.coordZ = float(self.coordZ)
        if self.CADprobability == None:
            self.CADprobability = 1.0
        else:
            self.CADprobability = float(self.CADprobability)
        self.diameter_mm = float(self.diameter_mm)

        return 1
