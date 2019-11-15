<<<<<<< HEAD
# -*- coding:utf-8 -*-
# utils used

import os
def check_path(_path):
    """Check weather the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
=======
# -*- coding:utf-8 -*-
# utils used

import os
def check_path(_path):
    """Check weather the _path exists. If not, make the dir."""
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
>>>>>>> a2587d06c292858265e54b24ac7ef835caccb239
            os.makedirs(os.path.dirname(_path))