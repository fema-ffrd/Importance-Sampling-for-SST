#region Distribution Helpers

#%% Function to pass parameters to truncnorm
def truncnorm_params(mean: float, std_dev: float, lower: float, upper: float) -> dict:
    '''
    Calculate the parameters for a truncated normal distribution.

    Args:
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
        lower (float): The lower bound of the truncated distribution.
        upper (float): The upper bound of the truncated distribution.

    Returns:
        dict: A dictionary containing the parameters 'a', 'b', 'loc', and 'scale' for the truncated normal distribution.

    '''
    d = dict(
        a = (lower - mean) / std_dev,
        b = (upper - mean) / std_dev,
        loc = mean,
        scale = std_dev
    )

    return d

#endregion -----------------------------------------------------------------------------------------
