#include <v2/coord.cuh>


/**
 * @brief why this class: need a unified and clear way to load memory between GMEM and SMEM
 * 
 * what this class do: 
 * 
 * - iterate through matrix data required by a tile in a col/row-majored array, fraction by fraction
 * 
 * - traverse a tile fraction and return data to caller one at a time (many at a time in future)
 * 
 * - get to next tile fraction, get to next responsible element, set correct fraction's boundary 
 * and start idx in a tile fraction
 * 
 * how:
 * 
 * - `ConstrainedCoord` uses tile fraction's boundary as constraints, and record thread's pointer coords
 * 
 * - `ConstrainedCoord` uses contiguous dimension of a matrix as inc_strided, and 1 as inc_contiguous 
 * (many at a time in future)
 * 
 * - `ConstrainedCoord` uses {start.contiguous + threadIdx.x % area.contiguous, 
 * start.strided + threadIdx.x / area.contiguous} as start coord
 * 
 * - set coord_tile_fraction and pointer in initialization
 * 
 * - set coord_tile_fraction and pointer, and add to constraints in next_tile_frac()
 * 
 * - add to pointer and coord_tile_fraction's current coord in next()
 * 
 * - return single element in getCurrentElement()
 */
template<typename T> class TileIterator
{
private:
    ConstrainedCoord coord_tile_fraction;
    Shape matrix_shape;

    T *array; 
    int index = 0;
    int inc_next_tile;
    int start_contiguous_in_tile;
    int start_strided_in_tile;
public:
    __device__ TileIterator(
        T *array, int mat_c, int mat_s,
        int bound_start_c, int bound_start_s,
        int bound_end_c, int bound_end_s,
        int start_c, int start_s,
        int inc_next_tile
    ):
        matrix_shape(mat_c, mat_s),
        inc_next_tile(inc_next_tile),
        start_contiguous_in_tile(start_c),
        start_strided_in_tile(start_s),
        array(array)
    {
        coord_tile_fraction.set_bounds(bound_start_c, bound_start_s, bound_end_c, bound_end_s);
        coord_tile_fraction.set_inc(1, mat_c);
        coord_tile_fraction.set(start_c + bound_start_c, start_s + bound_start_s);
        index = (bound_start_s + start_s) * mat_c + bound_start_c + start_c;
    }    

    __device__ ~TileIterator()
    {}    

    __device__ T getCurrentElement() 
    {
        return array[index];
    }

    __device__ void putElement(T e)
    {
        array[index] = e;
    }

    __device__ void next(int c, int s) 
    {
        index += coord_tile_fraction.next_contiguous(
                        s * matrix_shape.contiguous + c);
    }

    __device__ void next_tile()
    {
        int start_s = coord_tile_fraction.start.strided;
        int start_c = coord_tile_fraction.start.contiguous;
        int area_s = coord_tile_fraction.area.strided;
        int area_c = coord_tile_fraction.area.contiguous;

        coord_tile_fraction.set_bounds(
            start_c + inc_next_tile % matrix_shape.contiguous,
            start_s + inc_next_tile / matrix_shape.contiguous,
            start_c + inc_next_tile % matrix_shape.contiguous + area_c,
            start_s + inc_next_tile / matrix_shape.contiguous + area_s
        );

        start_s = coord_tile_fraction.start.strided;
        start_c = coord_tile_fraction.start.contiguous;
        coord_tile_fraction.set(start_c + start_contiguous_in_tile, 
                                start_s + start_strided_in_tile);

        index = (start_s + start_strided_in_tile) * matrix_shape.contiguous
                + start_c + start_contiguous_in_tile;
    }

    __device__ bool valid()
    {
        return coord_tile_fraction.valid() && (coord_tile_fraction.current < matrix_shape);
    }

    __device__ void reset()
    {
        int start_s = coord_tile_fraction.start.strided;
        int start_c = coord_tile_fraction.start.contiguous;
        coord_tile_fraction.set(start_contiguous_in_tile, start_strided_in_tile);
        index = (start_s + start_strided_in_tile) * matrix_shape.contiguous
                + start_c + start_contiguous_in_tile;
    }
};


