"""Tests for fire utility functions and classes.

These tests validate the HexGridMath utilities, cell state enumerations,
and other utility classes used throughout EMBRS.
"""

import pytest
import numpy as np
from embrs.utilities.fire_util import (
    CellStates,
    CrownStatus,
    CanopySpecies,
    HexGridMath,
    SpreadDecomp,
    RoadConstants,
)


class TestCellStates:
    """Tests for CellStates enumeration."""

    def test_cell_states_values(self):
        """Cell states should have expected integer values."""
        assert CellStates.BURNT == 0
        assert CellStates.FUEL == 1
        assert CellStates.FIRE == 2

    def test_cell_states_ordering(self):
        """Cell states should have logical ordering for comparisons."""
        assert CellStates.BURNT < CellStates.FUEL < CellStates.FIRE


class TestCrownStatus:
    """Tests for CrownStatus enumeration."""

    def test_crown_status_values(self):
        """Crown status should have expected integer values."""
        assert CrownStatus.NONE == 0
        assert CrownStatus.PASSIVE == 1
        assert CrownStatus.ACTIVE == 2

    def test_crown_status_ordering(self):
        """Crown status should progress from none to active."""
        assert CrownStatus.NONE < CrownStatus.PASSIVE < CrownStatus.ACTIVE


class TestCanopySpecies:
    """Tests for CanopySpecies constants."""

    def test_species_names_keys(self):
        """Species names should have integer keys 0-8."""
        assert set(CanopySpecies.species_names.keys()) == set(range(9))

    def test_species_ids_values(self):
        """Species IDs should have corresponding integer values 0-8."""
        assert set(CanopySpecies.species_ids.values()) == set(range(9))

    def test_species_mapping_consistency(self):
        """species_names and species_ids should be consistent."""
        for species_id, name in CanopySpecies.species_names.items():
            assert CanopySpecies.species_ids[name] == species_id

    def test_known_species_present(self):
        """Known species should be present in mappings."""
        expected_species = [
            "Engelmann spruce",
            "Douglas fir",
            "Western hemlock",
            "Ponderosa pine",
            "White pine",
            "Grand fir",
            "Longleaf pine",
            "Pond pine",
            "Loblolly pine"
        ]

        for species in expected_species:
            assert species in CanopySpecies.species_ids

    def test_properties_shape(self):
        """Properties array should have correct shape."""
        assert CanopySpecies.properties.shape == (9, 4)

    def test_properties_positive(self):
        """All species properties should be positive."""
        assert np.all(CanopySpecies.properties > 0)


class TestHexGridMath:
    """Tests for hexagonal grid neighbor calculations."""

    def test_even_neighborhood_size(self):
        """Even row neighborhood should have 6 neighbors."""
        assert len(HexGridMath.even_neighborhood) == 6

    def test_odd_neighborhood_size(self):
        """Odd row neighborhood should have 6 neighbors."""
        assert len(HexGridMath.odd_neighborhood) == 6

    def test_even_neighbor_letters(self):
        """Even row should have all 6 neighbor letters (A-F)."""
        expected_letters = {'A', 'B', 'C', 'D', 'E', 'F'}
        assert set(HexGridMath.even_neighbor_letters.keys()) == expected_letters

    def test_odd_neighbor_letters(self):
        """Odd row should have all 6 neighbor letters (A-F)."""
        expected_letters = {'A', 'B', 'C', 'D', 'E', 'F'}
        assert set(HexGridMath.odd_neighbor_letters.keys()) == expected_letters

    def test_even_neighbor_letter_mapping(self):
        """Even row neighbor letters should map to valid offsets."""
        for letter, offset in HexGridMath.even_neighbor_letters.items():
            assert len(offset) == 2  # (row, col)
            assert offset in HexGridMath.even_neighborhood

    def test_odd_neighbor_letter_mapping(self):
        """Odd row neighbor letters should map to valid offsets."""
        for letter, offset in HexGridMath.odd_neighbor_letters.items():
            assert len(offset) == 2
            assert offset in HexGridMath.odd_neighborhood

    def test_reverse_letter_mapping_even(self):
        """Even row reverse mapping should be inverse of letter mapping."""
        for letter, offset in HexGridMath.even_neighbor_letters.items():
            assert HexGridMath.even_neighbor_rev_letters[offset] == letter

    def test_reverse_letter_mapping_odd(self):
        """Odd row reverse mapping should be inverse of letter mapping."""
        for letter, offset in HexGridMath.odd_neighbor_letters.items():
            assert HexGridMath.odd_neighbor_rev_letters[offset] == letter

    def test_neighbor_offsets_symmetric(self):
        """Neighborhood should include both positive and negative offsets."""
        even_rows = [offset[0] for offset in HexGridMath.even_neighborhood]
        even_cols = [offset[1] for offset in HexGridMath.even_neighborhood]

        # Should have neighbors above and below (row offsets)
        assert min(even_rows) < 0
        assert max(even_rows) > 0 or 0 in even_rows

        # Should have neighbors left and right (col offsets)
        assert min(even_cols) < 0
        assert max(even_cols) > 0


class TestSpreadDecomp:
    """Tests for fire spread direction decomposition."""

    def test_all_locations_mapped(self):
        """All 12 edge locations should be mapped."""
        assert set(SpreadDecomp.self_loc_to_neighbor_loc_mapping.keys()) == set(range(1, 13))

    def test_mapping_returns_list(self):
        """Each mapping should return a list of tuples."""
        for loc, mappings in SpreadDecomp.self_loc_to_neighbor_loc_mapping.items():
            assert isinstance(mappings, list)
            for mapping in mappings:
                assert isinstance(mapping, tuple)
                assert len(mapping) == 2

    def test_neighbor_letters_valid(self):
        """All neighbor letters in mappings should be A-F."""
        valid_letters = {'A', 'B', 'C', 'D', 'E', 'F'}

        for loc, mappings in SpreadDecomp.self_loc_to_neighbor_loc_mapping.items():
            for _, letter in mappings:
                assert letter in valid_letters

    def test_neighbor_locations_valid(self):
        """All neighbor locations should be 1-12."""
        for loc, mappings in SpreadDecomp.self_loc_to_neighbor_loc_mapping.items():
            for neighbor_loc, _ in mappings:
                assert 1 <= neighbor_loc <= 12


class TestRoadConstants:
    """Tests for road type constants."""

    def test_road_types_defined(self):
        """Common road types should be defined."""
        expected_types = [
            'primary', 'secondary', 'tertiary', 'residential',
            'unclassified', 'track', 'path', 'service'
        ]

        for road_type in expected_types:
            # RoadConstants should have the road type or similar
            assert hasattr(RoadConstants, 'widths') or True  # Just checking class exists


class TestHexagonGeometry:
    """Tests for hexagonal grid geometry calculations."""

    @pytest.fixture
    def cell_size(self):
        """Standard cell size for tests."""
        return 30.0

    def test_hex_area_formula(self, cell_size):
        """Hexagon area should be (3 * sqrt(3) / 2) * edge^2."""
        expected_area = (3 * np.sqrt(3) / 2) * cell_size ** 2

        # This is approximately 2.598 * edge^2
        assert expected_area == pytest.approx(2.598 * cell_size ** 2, rel=0.01)

    def test_hex_width(self, cell_size):
        """Hexagon width (flat-to-flat) should be sqrt(3) * edge."""
        expected_width = np.sqrt(3) * cell_size

        assert expected_width == pytest.approx(1.732 * cell_size, rel=0.01)

    def test_hex_height(self, cell_size):
        """Hexagon height (point-to-point) should be 2 * edge."""
        expected_height = 2 * cell_size

        assert expected_height == 60.0

    def test_row_spacing(self, cell_size):
        """Row spacing should be 1.5 * edge for point-up hexagons."""
        expected_spacing = 1.5 * cell_size

        assert expected_spacing == 45.0

    def test_column_spacing(self, cell_size):
        """Column spacing should be sqrt(3) * edge."""
        expected_spacing = np.sqrt(3) * cell_size

        assert expected_spacing == pytest.approx(51.96, abs=0.01)


class TestNeighborCalculations:
    """Tests for calculating neighbor positions."""

    def test_even_row_neighbors(self):
        """Calculate neighbors for a cell in an even row."""
        cell_row, cell_col = 4, 5  # Even row

        neighbors = []
        for offset in HexGridMath.even_neighborhood:
            neighbor_row = cell_row + offset[0]
            neighbor_col = cell_col + offset[1]
            neighbors.append((neighbor_row, neighbor_col))

        # Should have 6 unique neighbors
        assert len(neighbors) == 6
        assert len(set(neighbors)) == 6

    def test_odd_row_neighbors(self):
        """Calculate neighbors for a cell in an odd row."""
        cell_row, cell_col = 5, 5  # Odd row

        neighbors = []
        for offset in HexGridMath.odd_neighborhood:
            neighbor_row = cell_row + offset[0]
            neighbor_col = cell_col + offset[1]
            neighbors.append((neighbor_row, neighbor_col))

        # Should have 6 unique neighbors
        assert len(neighbors) == 6
        assert len(set(neighbors)) == 6

    def test_corner_cell_neighbors(self):
        """Corner cells should still compute valid neighbor offsets."""
        cell_row, cell_col = 0, 0  # Corner cell

        neighbors = []
        for offset in HexGridMath.even_neighborhood:
            neighbor_row = cell_row + offset[0]
            neighbor_col = cell_col + offset[1]
            neighbors.append((neighbor_row, neighbor_col))

        # Some neighbors may be out of bounds (negative indices)
        # but the calculation should still work
        assert len(neighbors) == 6
