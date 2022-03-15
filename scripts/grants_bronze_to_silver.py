# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def bronze_to_silver(verbose: bool = True) -> None:
    """Process the grants dataset.
        1. Drop duplicates
        2. Create UUIDs
        3. Replace missing EINs w/ UUIDs
    Args:
        verbose (bool, optional): Print statements for debugging. Defaults to True.
    """
    import uuid

    # Read in grants data
    grants_df = pd.read_csv(f"{BRONZE_PATH}2010_to_2016_grants.csv").drop_duplicates()
    if verbose:
        print(f"Grants data: {grants_df.shape[0]:,}")

    # Convert EIN, Zip codes to strings
    num_cols = ["granteeein", "grantorein", "granteezipcode", "grantorzipcode"]
    grants_df.loc[:, num_cols] = (
        grants_df[num_cols].fillna(0).astype(int).astype(str).replace("0", pd.NA)
    )

    # Make sure ZIP codes have 5 numbers
    for col in ["grantorzipcode", "granteezipcode"]:
        grants_df.loc[:, col] = (
            grants_df[col]
            .apply(lambda val: f"{'0'*(5-len(val))}{val}" if not pd.isna(val) else None)
            .replace("00000", pd.NA)
            .fillna(pd.NA)
        )

    # Convert Grant amount to integer then to strings
    grants_df.loc[:, "cashgrantamt"] = (
        grants_df["cashgrantamt"]
        .fillna(999999)
        .astype(int)
        .astype(str)
        .replace("999999", pd.NA)
    )

    # Convert tax period to string
    grants_df.loc[:, "taxperiod"] = grants_df["taxperiod"].astype(str)

    # Replace all other NaN values with pd.NA
    grants_df.loc[:, :] = grants_df.fillna(pd.NA)

    # Combining location fields (city, state, zipcode) for de-duping
    for org_type in ["grantor", "grantee"]:
        loc_cols = [f"{org_type}city", f"{org_type}state", f"{org_type}zipcode"]
        grants_df.loc[:, f"{org_type}_location"] = (
            grants_df[loc_cols]
            .apply(lambda x: x.str.cat(sep=","), axis=1)
            .replace("", pd.NA)
        )

    # Combining name and location for de-duping
    for org_type in ["grantor", "grantee"]:
        combined_cols = [org_type, f"{org_type}_location"]
        grants_df.loc[:, f"{org_type}_info"] = (
            grants_df[combined_cols]
            .apply(lambda x: x.str.cat(sep=","), axis=1)
            .replace("", pd.NA)
        )

    # Create new UUID for all unique grantee_info
    noms = grants_df.loc[
        grants_df["granteeein"].isna() & grants_df["grantee_info"].notna(),
        "grantee_info",
    ].unique()
    noms_mapper = {nom: uuid.uuid4() for nom in noms}
    grants_df.loc[
        grants_df["granteeein"].isna() & grants_df["grantee_info"].notna(), "granteeein"
    ] = grants_df["grantee_info"].map(noms_mapper)

    # Create new UUID for all NaN grantee fields
    indexes = grants_df.loc[grants_df["granteeein"].isna(), :].index
    unique_uuids = [str(uuid.uuid4()) for idx in indexes]
    grants_df.loc[indexes, "granteeein"] = unique_uuids

    save_to_parquet(
        data=grants_df[GRANTS_HEADERS],
        cols=GRANTS_HEADERS,
        loc=GRANTS_SILVER_PATH,
        filename="grants",
    )


def main():
    """
    Create dataset for the model, saving to parquet for easy access.
    """
    return bronze_to_silver(verbose=True)


if __name__ == "__main__":
    main()
