#ifndef __RATIONAL_H
#define __RATIONAL_H

typedef struct {
	long numerator;
	long denominator;
} rational_t;

#define RATIONAL_NAN { 0, 0 };

 rational_t rational_init(double x);
__device__ rational_t rational_negate(rational_t x);
__device__ rational_t rational_add(rational_t x, rational_t y);
__device__ rational_t rational_subtract(rational_t x, rational_t y);
__device__ rational_t rational_multiply(rational_t x, rational_t y);
__device__ rational_t rational_divide(rational_t x, rational_t y);
 double get_value(rational_t x);

#endif