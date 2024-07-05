from httpx import HTTPStatusError
from tenacity import retry as _retry, stop_after_attempt, wait_random_exponential


def should_retry(retry_state):
    if not (exception := retry_state.outcome.exception()):
        return False

    if isinstance(exception, HTTPStatusError):
        return exception.response.status_code // 100 > 3

    return True


def get_retry_wrapper(n_retries: int = 3):
    return _retry(
        stop=stop_after_attempt(n_retries),
        wait=wait_random_exponential(max=10),
        retry=should_retry,
        reraise=True,
    )
