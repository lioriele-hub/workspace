
import re
import os
import pandas as pd
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_test', action='store_true', help='Set test mode (default False)')
    args = parser.parse_args()

    print(re.match('a', 'abc'))
    print('hello')
    print('IS_TEST env:', os.environ.get('IS_TEST', 'Not Set'))
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    print(df)

    # CLI flag takes precedence; fall back to environment variable values: 1/yes/true
    env_is_test = str(os.environ.get('IS_TEST', '')).lower() in ('1', 'yes', 'true')
    effective_is_test = args.is_test or env_is_test
    print(f'is_test: {effective_is_test}')