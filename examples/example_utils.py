import pandas as pd

VIX = "PX_OPEN_VIX_volatility"

def add_AR_cols(df: pd.DataFrame, lags: int, return_cols=False)-> pd.DataFrame:
    """
    added columns defined by VIX_lagged_i = VIX - VIX_-i
    :param df: DataFrame containing all variables
    :param lags: number of passed unit time added to the DataFrame
    :return: the DataFrame with the lagged columns
    """
    VIX = 'PX_OPEN_VIX_volatility'
    cols=[]

    for i in range(1,lags):
        df['VIX_LAG_' + str(i)] = df[VIX] - df[VIX].shift(i)
        cols.append('VIX_LAG_' + str(i))


    if return_cols: return df.dropna(), cols
    else: return df.dropna()



def create_box(df: pd.DataFrame, threshold=2., box_length=7, relative_threshold=None) -> pd.DataFrame:

    if relative_threshold:
        for i in range(-1, -box_length, -1):
            # Est-ce que le i-ème jour est au dessus du threshold haut ?
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[VIX] * (1. + relative_threshold))
            df["VIX_inf_" + str(i)] = (df[VIX].shift(i) < df[VIX] * (1. - relative_threshold))

    else:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[
                VIX] + threshold)  # Est-ce que le i-ème jour est au dessus du threshold haut ?
            df["VIX_inf_" + str(i)] = (df[VIX].shift(i) < df[
                VIX] - threshold)  # Est-ce que le i-ème jour est en dessous du threshold bas ?

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite
    # (0) ?
    df["Box"] = 1 * df["VIX_sup_-1"] + (-1) * df["VIX_inf_-1"]

    for i in range(-2, -box_length, -1):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du
        # tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 * df[f"VIX_sup_{i}"] + (-1) * df[f"VIX_inf_{i}"]) + (df["Box"] != 0) * df[
            "Box"]

    df = df.drop(
        columns=["VIX_sup_" + str(i) for i in range(box_length)] + ["VIX_inf_" + str(i) for i in range(-1, -box_length, -1)])
    return df


def create_binary_box(df: pd.DataFrame, threshold=2., box_length=7, relative_threshold = None) -> pd.DataFrame:

    if relative_threshold:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[VIX] *(1+relative_threshold))

    else:
        for i in range(-1, -box_length, -1):
            df["VIX_sup_" + str(i)] = (df[VIX].shift(i) > df[
                VIX] + threshold)  # Est-ce que le i-ème jour est au dessus du threshold haut ?

    # Initialisation : Est-ce que le dès le lendemain, on dépasse (1), on est en dessous (-1) ou on est dans la boite
    # (0) ?
    df["Box"] = 1 * df["VIX_sup_-1"]

    for i in range(-1, -box_length, -1):
        # Iterations : Si la valeur n'est pas encore sortie (Box == 0), sort-elle par le haut (1), bas (-1) ou pas du
        # tout (0) de la boite au jour i ?
        df["Box"] = (df["Box"] == 0) * (1 * df[f"VIX_sup_{i}"]) + (df["Box"] != 0) * df["Box"]

    df = df.drop(columns=["VIX_sup_" + str(i) for i in range(-1, -box_length, -1)])
    return df