
import re
import os
import pandas as pd
if __name__ == '__main__':
    print(re.match('a', 'abc'))
    print('hello')
    print(os.environ.get('IS_TEST', 'Not Set'))
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    print(df)