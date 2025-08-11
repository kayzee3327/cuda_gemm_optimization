#pragma once


class Coord
{
private:

public:
    int contiguous;
    int strided;

    __device__ Coord (int c, int s):
        contiguous(c),
        strided(s)
    {}

    __device__ Coord () {}

    __device__ bool operator<(Coord a)
    {
        return contiguous < a.contiguous && strided < a.strided;
    }

    __device__ bool operator==(Coord a)
    {
        return contiguous == a.contiguous && strided == a.strided;
    }
    
    __device__ bool operator>(Coord a)
    {
        return contiguous > a.contiguous && strided > a.strided;
    }

    __device__ bool operator>=(Coord a) 
    {
        return contiguous >= a.contiguous && strided >= a.strided;
    }

    __device__ bool operator<=(Coord a) 
    {
        return contiguous <= a.contiguous && strided <= a.strided;
    }

    __device__ bool operator!=(Coord a)
    {
        return contiguous != a.contiguous || strided != a.strided;
    }
};

using Shape = Coord;


class ConstrainedCoord
{
public:
    // 2D area for allowed coords [start, end)
    Coord start;
    Coord end;
    Coord current;

    // Shape of this allowed area
    Shape area;

    int inc_contiguous; // steps for add 1 contiguous
    int inc_strided;    // steps for add 1 strided

    __device__ ConstrainedCoord() {}

    __device__ void set(int c, int s)
    {
        current.contiguous = c;
        current.strided = s;
    }

    __device__ void set_inc(int c, int s)
    {
        inc_contiguous = c;
        inc_strided = s;
    }

    __device__ void set_bounds(
        int start_c, int start_s,
        int end_c, int end_s
    )
    {
        start.contiguous = start_c;
        start.strided = start_s;
        end.contiguous = end_c;
        end.strided = end_s;
        area.contiguous = end_c - start_c;
        area.strided = end_s - start_s;
    }

    __device__ bool valid()
    {
        return current >= start && current < end;
    }
    
    // return how many steps it moved
    __device__ int next_contiguous(int inc)
    {
        int inc_c = inc % inc_strided;
        int inc_s = inc / inc_strided;
        current.contiguous += inc_c;
        current.strided += inc_s;
        return inc_c * inc_contiguous + inc_s * inc_strided;
    }

    // return how many steps it moved
    __device__ int next_strided(int inc)
    {
        current.strided += inc;
        return inc * inc_strided;
    }
};