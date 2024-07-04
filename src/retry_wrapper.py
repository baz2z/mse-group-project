from tenacity import retry as _retry, stop_after_attempt, wait_random_exponential


def get_retry_wrapper(logger, n_retries=3):
    def before_sleep(retry_state):
        logger.error(f"Retrying after {retry_state.outcome.exception()}")

    return _retry(
        stop=stop_after_attempt(n_retries),
        wait=wait_random_exponential(),
        before_sleep=lambda retry_state: before_sleep(retry_state),
    )
