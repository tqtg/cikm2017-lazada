import os
import numpy as np

def write_submission(filename, predicted_results):
    if not os.path.exists('submission'):
        os.makedirs('submission')
    np.savetxt('submission/' + filename, predicted_results, fmt='%.5f')
    print(filename + ' updated!')