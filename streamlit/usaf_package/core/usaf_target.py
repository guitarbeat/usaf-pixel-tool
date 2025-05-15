class USAFTarget:
    """
    USAF 1951 Resolution Target definitions and calculations.

    Provides formulas for line pairs per mm and line pair width in microns
    according to MIL-STD-150A.
    """

    @classmethod
    def lp_per_mm(cls, group: int, element: int) -> float:
        """
        Calculate the line pairs per millimeter (lp/mm) for a given group and element.

        Args:
            group (int): USAF group number
            element (int): USAF element number
        Returns:
            float: Line pairs per millimeter
        """
        return 2 ** (group + (element - 1) / 6)

    @classmethod
    def line_pair_width_microns(cls, group: int, element: int) -> float:
        """
        Calculate the line pair width in microns for a given group and element.

        Args:
            group (int): USAF group number
            element (int): USAF element number
        Returns:
            float: Line pair width in microns
        """
        lpmm = cls.lp_per_mm(group, element)
        return 1000.0 / (2 * lpmm)
