# Copyright (c) 2025-2026, rocQuantum Developers.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Concrete implementation of a decoder for the 3-qubit bit-flip repetition code.
"""

from typing import List

from rocquantum.python.rocq import PauliOperator
from rocquantum.qec.framework import Decoder

class RepetitionCodeDecoder(Decoder):
    """
    A simple lookup-table decoder for the 3-qubit repetition code.
    """
    def decode(self, syndrome: List[int]) -> PauliOperator:
        """
        Decodes the 2-bit syndrome to find the location of a single X error.
        Syndrome bits correspond to [Z0Z1, Z1Z2] measurements.
        """
        if syndrome == [0, 0]:
            # No error detected
            return PauliOperator()
        elif syndrome == [1, 0]:
            # Error on data qubit 0
            return PauliOperator({"X0": 1.0})
        elif syndrome == [1, 1]:
            # Error on data qubit 1
            return PauliOperator({"X1": 1.0})
        elif syndrome == [0, 1]:
            # Error on data qubit 2
            return PauliOperator({"X2": 1.0})
        else:
            # This case implies more than one error, which this simple
            # code cannot correct. Return no correction.
            return PauliOperator()
