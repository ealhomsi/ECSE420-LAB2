#include "rational.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

__device__ long gcd(long a, long b)
{
	long temp;

	while (b != 0)
	{
		temp = a % b;

		a = b;
		b = temp;
	}

	return a;
}

__device__ double dabs(double x)
{
	return x > 0 ? x : -x;
}

__device__ rational_t rational_scale(rational_t r, int scale, int inverse)
{
	if (inverse)
	{
		return  {
			r.numerator / scale,
			r.denominator / scale
		};
	}

	return {
		r.numerator * scale,
		r.denominator * scale
	};
}

__device__ rational_t rational_reduced_form(long numerator, long denominator)
{
	long scale = gcd(numerator, denominator);
	rational_t r = {
		numerator,
		denominator
	};

	return rational_scale(r, scale, 1);
}

__device__ rational_t rational_init(double x)
{
	long variable = 1;
	while (dabs(round(x) - x) > DBL_EPSILON)
	{
		x *= 10;
		variable *= 10;
	}

	return rational_reduced_form(
		(long)round(x),
		variable);
}

__device__ rational_t rational_negate(rational_t x)
{
	return {
		-x.numerator,
			x.denominator
	};
}

__device__ rational_t rational_add(rational_t x, rational_t y)
{
	if (y.denominator == x.denominator)
	{
		return rational_reduced_form(x.numerator + y.numerator, x.denominator);
	}

	rational_t xScaled = rational_scale(x, y.denominator, 0);
	rational_t yScaled = rational_scale(y, x.denominator, 0);

	return rational_reduced_form(
		xScaled.numerator + yScaled.numerator,
		xScaled.denominator);
}

__device__ rational_t rational_subtract(rational_t x, rational_t y)
{
	return rational_add(x, rational_negate(y));
}

__device__ rational_t rational_multiply(rational_t x, rational_t y)
{
	return rational_reduced_form(
		x.numerator * y.numerator,
		x.denominator * y.denominator);
}

__device__ rational_t rational_divide(rational_t x, rational_t y)
{
	// Assume denominator is always nonzero
	if (y.numerator == 0)
	{
		return RATIONAL_NAN;
	}

	return rational_reduced_form(
		x.numerator * y.denominator,
		x.denominator * y.numerator);
}

__device__ double get_value(rational_t x) {
	return (1.0 * x.numerator) / x.denominator;
}