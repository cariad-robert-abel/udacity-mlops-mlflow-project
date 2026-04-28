#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data: pd.DataFrame):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.to_numpy()  # Using to_numpy() for better performance

    # This also enforces the same order
    assert np.array_equal(expected_colums, these_columns)  # Using numpy comparison for better performance


def test_neighborhood_names(data: pd.DataFrame):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################

def test_row_count(data: pd.DataFrame):
    """Test for sensible number of rows in batched CSV data"""
    assert 15000 < data.shape[0] < 1_000_000, "Dataframe should have between 15k and 1M rows"

def test_price_range(data: pd.DataFrame, min_price: int, max_price: int):
    """Test that all prices are between min_price and max_price"""
    assert data['price'].between(min_price, max_price).all(), f"All prices should be between {min_price} and {max_price}"
