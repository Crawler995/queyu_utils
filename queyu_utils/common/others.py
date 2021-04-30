import time


def get_cur_time_str() -> str:
    """Get string represented current timestamp.

    Returns:
        String represented current timestamp.
    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
