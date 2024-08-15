Installation Guide
==================

.. note::
    `AutoMLQuantILDetect` with package ``autoqild`` is intended to work with **Python 3.9.5 and above**.

Installation Steps
------------------

1. **Clone the Repository**:

    .. code-block:: sh

        git clone https://github.com/LeakDetectAI/AutoMLQuantILDetect.git
        cd AutoMLQuantILDetect

2. **Create and Activate a Conda Environment**:

    .. code-block:: sh

        conda create --name ILD python=3.10
        conda activate ILD
3. **Installation**:

    .. code-block:: sh

            export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

    .. code-block:: sh

            pip install -r requirements.txt

    - **OR**

    .. code-block:: sh

            python setup.py install