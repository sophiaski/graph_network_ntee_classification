# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def clean_grants(verbose: bool = True) -> pd.DataFrame:
    """Process the grants dataset:
        1. Drop duplicates
        2. Create UUIDs
        3. Replace missing EINs w/ UUIDs (Based on uniquenness of name and location information)
    
    Args:
        verbose (bool, optional): Print statements for debugging. Defaults to True.

    Returns:
        pd.DataFrame: Grants DataFrame with the specified columns in the GRANTS_HEADERS object.
    """
    # To fill in missing EINs
    from uuid import uuid4

    # To help with processing missing data rows
    FILLER = 999999999

    # Read in dataset CSV
    grants = pd.read_csv(f"{BRONZE_PATH}2010_to_2016_grants.csv").drop_duplicates()
    if verbose:
        print(f"Grants data: {grants.shape[0]:,}")

    # Convert EIN and ZIP codes to strings
    num_cols = ["granteeein", "grantorein", "granteezipcode", "grantorzipcode"]
    grants.loc[:, num_cols] = (
        grants[num_cols]
        .fillna(FILLER)
        .astype(int)
        .astype(str)
        .replace(str(FILLER), pd.NA)
    )

    # Make sure EINs have 9 number
    for col in ["granteeein", "grantorein"]:
        grants.loc[:, col] = (
            grants[col]
            .apply(lambda val: f"{'0'*(9-len(val))}{val}" if not pd.isna(val) else None)
            .replace("000000000", pd.NA)
            .fillna(pd.NA)
        )

    # Make sure ZIP codes have 5 numbers
    for col in ["grantorzipcode", "granteezipcode"]:
        grants.loc[:, col] = (
            grants[col]
            .apply(lambda val: f"{'0'*(5-len(val))}{val}" if not pd.isna(val) else None)
            .replace("00000", pd.NA)
            .fillna(pd.NA)
        )

    # Convert grant amount to integer then to strings
    grants.loc[:, "cashgrantamt"] = (
        grants["cashgrantamt"]
        .fillna(FILLER)
        .astype(int)
        .astype(str)
        .replace(str(FILLER), pd.NA)
    )

    # Convert tax period to string
    grants.loc[:, "taxperiod"] = grants["taxperiod"].astype(str)

    # Replace all other missing values with pd.NA
    grants.loc[:, :] = grants.fillna(pd.NA)

    # Load clean up dictionaries and apply to fields where there are missing values
    cleanup_dir = f"{BRONZE_PATH}cleanup/"
    for org_type in ["grantor", "grantee"]:
        for val1, val2 in [
            ("CITY", "city"),
            ("NAME", ""),
            ("STATE", "state"),
            ("ZIP5", "zipcode"),
        ]:
            mapper = load_json(loc=cleanup_dir, filename=f"ein_{val1}")
            grants.loc[grants[f"{org_type}{val2}"].isna(), f"{org_type}{val2}"] = (
                grants[grants[f"{org_type}{val2}"].isna()][f"{org_type}ein"]
                .map(mapper)
                .fillna(pd.NA)
            )

    # Combining location fields (city, state, zipcode) for de-duping
    for org_type in ["grantor", "grantee"]:
        loc_cols = [f"{org_type}city", f"{org_type}state", f"{org_type}zipcode"]
        grants.loc[:, f"{org_type}_location"] = (
            grants[loc_cols]
            .apply(lambda x: x.str.cat(sep=","), axis=1)
            .replace("", pd.NA)
        )

    # Combining name and location for de-duping
    for org_type in ["grantor", "grantee"]:
        combined_cols = [org_type, f"{org_type}_location"]
        grants.loc[:, f"{org_type}_info"] = (
            grants[combined_cols]
            .apply(lambda x: x.str.cat(sep=","), axis=1)
            .replace("", pd.NA)
        )
    # Create new UUID for all unique grantee_info
    noms = grants.loc[
        grants["granteeein"].isna() & grants["grantee_info"].notna(), "grantee_info",
    ].unique()
    noms_mapper = {nom: str(uuid4()) for nom in noms}
    grants.loc[
        grants["granteeein"].isna() & grants["grantee_info"].notna(), "granteeein"
    ] = (grants["grantee_info"].map(noms_mapper).fillna(pd.NA))

    # Create new UUID for all NA grantee fields
    indexes = grants.loc[grants["granteeein"].isna(), :].index
    unique_uuids = [str(uuid4()) for idx in indexes]
    grants.loc[indexes, "granteeein"] = unique_uuids

    # Switch NAs to empty strings to conform to pyarrow schema
    grants.loc[:, :] = grants.fillna("")

    # Rename columns to more pythonic formatting
    grants.rename(
        {
            "granteeein": "grantee_ein",
            "grantdesc": "grant_desc",
            "cashgrantamt": "cash_grant_amt",
            "grantorein": "grantor_ein",
            "taxperiod": "tax_period",
            "granteecity": "grantee_city",
            "granteestate": "grantee_state",
            "granteezipcode": "grantee_zipcode",
            "grantorcity": "grantor_city",
            "grantorstate": "grantor_state",
            "grantorzipcode": "grantor_zipcode",
        },
        axis=1,
        inplace=True,
    )

    return grants[GRANTS_HEADERS]


@typechecked
def load_grants(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Load in the cleaned grants data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame.
    """
    # Load in data
    loc = GRANTS_SILVER_PATH
    filename = "grants"

    # Return dataframe with columns sorted alphabetically
    return load_parquet(
        loc=loc, filename=filename, frac=frac, seed=seed, verbose=verbose
    )


def main():
    grants = clean_grants(verbose=True)
    save_to_parquet(
        data=grants[GRANTS_HEADERS],
        cols=GRANTS_HEADERS,
        loc=GRANTS_SILVER_PATH,
        filename="grants",
    )


if __name__ == "__main__":
    main()
