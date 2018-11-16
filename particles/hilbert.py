# -*- coding: utf-8 -*-

""" Hilbert curve and its inverse, in any dimension. 

"""

import numpy as np
from numba import jit
from numba.types import int64


def invlogit(x):
    return 1. / (1. + np.exp(-x))


@jit(nopython=True)
def hilbert_array(xint):
    """Compute Hilbert indices. 

    Parameters
    ----------
    xint: (N, d) int numpy.ndarray

    Returns
    -------
    h: (N,) int numpy.ndarray
        Hilbert indices 
    """
    N, d = xint.shape
    h = np.zeros(N, int64)
    for n in range(N):
        h[n] = Hilbert_to_int(xint[n, :])
    return h


@jit()
def hilbert_sort(x):
    """Hilbert sort: sort vectors according to their Hilbert index. 
    
    Parameters
    ----------
    x : (N,) or (N, d) float numpy.ndarray
        N vectors in R^d

    Returns
    -------
    A : (N,) int numpy.ndarray
        argsort (e.g. x[A[0], :] is the vector with smallest H index). 

    Note
    ----
    The vectors in `x` are first (a) standardized; and (b) applied a logistic
    transformation so as to lie in [0,1]^d. 
    """
    d = 1 if x.ndim == 1 else x.shape[1]
    if d == 1:
        return np.argsort(x, axis=0)
    xs = invlogit((x - np.mean(x, axis=0)) / np.std(x, axis=0))
    maxint = np.floor(2**(62 / d))
    xint = np.floor(xs * maxint).astype('int')
    return np.argsort(hilbert_array(xint))


# hilbert.py -- Hilbert walk coordinate codec in multiple dimensions.
#    int_to_Hilbert( i, 3 ) ==> ( x, y, z )
#        int_to_Hilbert( 0, nD ) ==> ( 0, 0, 0, ... 0 ) Start at origin.
#        int_to_Hilbert( 1, nD ) ==> ( 1, 0, 0, ... 0 ) 1st step is along x.
#    Hilbert_to_int( ( x, y, z ) ) ==> i
# Steve Witham ess doubleyou at tiac remove-this dot net.
# http://www.tiac.net/~sw/2008/10/Hilbert


@jit(nopython=True)
def int_to_Hilbert(i, nD=2):  # Default is the 2D Hilbert walk.
    index_chunks = unpack_index(i, nD)
    nChunks = len(index_chunks)
    mask = 2 ** nD - 1
    start, end = initial_start_end(nChunks, nD)
    coord_chunks = np.zeros(nChunks, int64)
    for j in range(nChunks):
        i = index_chunks[j]
        coord_chunks[j] = gray_encode_travel(start, end, mask, i)
        start, end = child_start_end(start, end, mask, i)
    return pack_coords(coord_chunks, nD)


@jit(nopython=True)
def Hilbert_to_int(coords):
    nD = coords.shape[0]
    coord_chunks = unpack_coords(coords)
    nChunks = len(coord_chunks)
    mask = 2 ** nD - 1
    start, end = initial_start_end(nChunks, nD)
    index_chunks = np.zeros(nChunks, int64)
    for j in range(nChunks):
        i = gray_decode_travel(start, end, mask, coord_chunks[j])
        index_chunks[j] = i
        start, end = child_start_end(start, end, mask, i)
    return pack_index(index_chunks, nD)


@jit(nopython=True)
def initial_start_end(nChunks, nD):
    # This orients the largest cube so that
    # its start is the origin (0 corner), and
    # the first step is along the x axis, regardless of nD and nChunks:
    return 0, 2**((-nChunks - 1) % nD)  # in Python 0 <=  a % b  < b.


# Unpacking arguments and packing results of int <-> Hilbert functions.
# nD == # of dimensions.
# A "chunk" is an nD-bit int (or Python long, aka bignum).
# Lists of chunks are highest-order first.
# Bits within "coord chunks" are x highest-order, y next, etc.,
# i.e., the same order as coordinates input to Hilbert_to_int()
# and output from int_to_Hilbert().


# unpack_index( int index, nD ) --> list of index chunks.
#

@jit(nopython=True)
def unpack_index(i, nD):
    p = 2**nD     # Chunks are like digits in base 2**nD.

    # numpy doesn't have logs of arbitrary base, have to do this instead:
    nChunks = int(np.ceil(np.lop(i + 1) / np.log(p)))

    if nChunks < 1:
        nChunks = 1

    # nChunks = max(1, int(ceil(log(i + 1, p))))  # of digits
    chunks = np.zeros(nChunks, dtype=int64)
    for j in range(nChunks - 1, -1, -1):
        chunks[j] = i % p
        i /= p
    return chunks


@jit(nopython=True)
def pack_index(chunks, nD):
    p = 2**nD  # Turn digits mod 2**nD back into a single number:
    z = chunks[0]
    for i in range(1, len(chunks)):
        z = p * z + chunks[i]
    return z


# unpack_coords( list of nD coords ) --> list of coord chunks each nD bits.

@jit(nopython=True)
def unpack_coords(coords):
    nD = coords.shape[0]
    biggest = np.max(coords)  # the max of all coords

    nChunks = int(np.ceil(np.log2(biggest + 1)))

    if nChunks < 1:
        nChunks = 1

    # nChunks2 = max(1, int(ceil(log(biggest + 1, 2))))  # max # of bits

    #print (nChunks, nChunks2)
    return transpose_bits(coords, nChunks)


@jit(nopython=True)
def pack_coords(chunks, nD):
    return transpose_bits(chunks, nD)


# transpose_bits --
#    Given nSrcs source ints each nDests bits long,
#    return nDests ints each nSrcs bits long.
#    Like a matrix transpose where ints are rows and bits are columns.
#    Earlier srcs become higher bits in dests;
#    earlier dests come from higher bits of srcs.

@jit(nopython=True)
def transpose_bits(srcs, nDests):
    srcs = srcs.copy()  # Make a copy we can modify safely.
    nSrcs = srcs.shape[0]
    dests = np.zeros((nDests,), dtype=int64)
    #dests = np.zeros(nDests, dtype='int')
    # Break srcs down least-significant bit first, shifting down:
    for j in range(nDests - 1, -1, -1):
        # Put dests together most-significant first, shifting up:
        dest = 0
        for k in range(nSrcs):
            dest = dest * 2 + srcs[k] % 2
            srcs[k] /= 2
        dests[j] = dest
    return dests


# Gray encoder and decoder from http://en.wikipedia.org/wiki/Gray_code :
#
@jit(nopython=True)
def gray_encode(bn):
    assert bn >= 0
    #assert type( bn ) in [ int, long ]

    return bn ^ (bn // 2)


@jit(nopython=True)
def gray_decode(n):
    sh = 1
    while True:
        div = n >> sh
        n ^= div
        if div <= 1:
            return n

        sh <<= 1


# gray_encode_travel -- gray_encode given start and end using bit rotation.
#    Modified Gray code.  mask is 2**nbits - 1, the highest i value, so
#        gray_encode_travel( start, end, mask, 0 )    == start
#        gray_encode_travel( start, end, mask, mask ) == end
#        with a Gray-code-like walk in between.
#    This method takes the canonical Gray code, rotates the output word bits,
#    then xors ("^" in Python) with the start value.
#
@jit(nopython=True)
def gray_encode_travel(start, end, mask, i):
    travel_bit = start ^ end
    modulus = mask + 1          # == 2**nBits
    # travel_bit = 2**p, the bit we want to travel.
    # Canonical Gray code travels the top bit, 2**(nBits-1).
    # So we need to rotate by ( p - (nBits-1) ) == (p + 1) mod nBits.
    # We rotate by multiplying and dividing by powers of two:
    g = gray_encode(i) * (travel_bit * 2)
    return ((g | (g // modulus)) & mask) ^ start


@jit(nopython=True)
def gray_decode_travel(start, end, mask, g):
    travel_bit = start ^ end
    modulus = mask + 1          # == 2**nBits
    rg = (g ^ start) * (modulus // (travel_bit * 2))
    return gray_decode((rg | (rg // modulus)) & mask)


# child_start_end( parent_start, parent_end, mask, i ) -- Get start & end for child.
#    i is the parent's step number, between 0 and mask.
#    Say that parent( i ) =
#           gray_encode_travel( parent_start, parent_end, mask, i ).
#    And child_start(i) and child_end(i) are what child_start_end()
#    should return -- the corners the child should travel between
#    while the parent is in this quadrant or child cube.
#      o  child_start( 0 ) == parent( 0 )       (start in a corner)
#      o  child_end( mask ) == parent( mask )   (end in a corner)
#      o  child_end(i) - child_start(i+1) == parent(i+1) - parent(i)
#         (when parent bit flips, same bit of child flips the opposite way)
#    Those constraints still leave choices when nD (# of bits in mask) > 2.
#    Here is how we resolve them when nD == 3 (mask == 111 binary),
#    for parent_start = 000 and parent_end = 100 (canonical Gray code):
#         i   parent(i)    child_
#         0     000        000   start(0)    = parent(0)
#                          001   end(0)                   = parent(1)
#                 ^ (flip)   v
#         1     001        000   start(1)    = parent(0)
#                          010   end(1)                   = parent(3)
#                ^          v
#         2     011        000   start(2)    = parent(0)
#                          010   end(2)                   = parent(3)
#                 v          ^
#         3     010        011   start(3)    = parent(2)
#                          111   end(3)                   = parent(5)
#               ^          v
#         4     110        011   start(4)    = parent(2)
#                          111   end(4)                   = parent(5)
#                 ^          v
#         5     111        110   start(5)    = parent(4)
#                          100   end(5)                   = parent(7)
#                v          ^
#         6     101        110   start(6)    = parent(4)
#                          100   end(6)                   = parent(7)
#                 v          ^
#         7     100        101   start(7)    = parent(6)
#                          100   end(7)                   = parent(7)
#    This pattern relies on the fact that gray_encode_travel()
#    always flips the same bit on the first, third, fifth, ... and last flip.
#    The pattern works for any nD >= 1.
#
@jit(nopython=True)
def child_start_end(parent_start, parent_end, mask, i):
    start_i = max(0, (i - 1) & ~1)  # next lower even number, or 0
    end_i = min(mask, (i + 1) | 1)  # next higher odd number, or mask
    child_start = gray_encode_travel(parent_start, parent_end, mask, start_i)
    child_end = gray_encode_travel(parent_start, parent_end, mask, end_i)
    return child_start, child_end
