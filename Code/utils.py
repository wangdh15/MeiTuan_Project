# -*- coding:utf-8 -*-
# utils used

import os
def check_path(_path):
    """Check weather the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))