
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// core/microfacet.cpp*
#include "microfacet.h"
#include "reflection.h"
#include "rng.h"

namespace pbrt {

static RNG rng;

Float FlatGaussianElementsDistribution::getFlatGaussian2DConstant(Float c, Vector2f s, Float invSigmaRSq) const {
	return c * std::exp(-0.5 * invSigmaRSq * s.LengthSquared());
}

Float FlatGaussianElementsDistribution::evaluate2DFlatGaussian(Float c, Vector2f u, Vector2f u0, Float invCov) const {
	return c * std::exp(-0.5 * (invCov * (u - u0).LengthSquared()));
}

Float FlatGaussianElementsDistribution::getFlatGaussianProductCov(Float invSigmaHSq, Float invFootprintCov) const {
	return 1.f / (invSigmaHSq + invFootprintCov);
}

Vector2f FlatGaussianElementsDistribution::getFlatGaussianProductMean(Float finalCov, Float invCov1, Float invCov2, Vector2f mu1, Vector2f mu2) const {
	return finalCov * (invCov1 * mu1 + invCov2 * mu2);
}

Float FlatGaussianElementsDistribution::getFlatGaussianProductScalingCoeff(Vector2f finalMu, Float c1, Float c2, Vector2f mu1, Vector2f mu2,
	Float invCov1, Float invCov2) const {
	return evaluate2DFlatGaussian(c1, finalMu, mu1, invCov1) * evaluate2DFlatGaussian(c2, finalMu, mu2, invCov2);
}

Float FlatGaussianElementsDistribution::evaluateFlatPNDF(Float c, Vector2f u, Vector2f s, Float invSigmaHSq, Float invSigmaRSq,
	Vector2f footprintMean, Float invFootprintCov) const {

	Vector2f u0(0, 0);   // Flat Gaussian has a diagonal matrix for invCov
	Float c1 = getFlatGaussian2DConstant(c, s, invSigmaRSq);
	Float c2 = 1.f / (2 * Pi * (1.f / invFootprintCov));    // normalizing the integration
	Float finalCov = getFlatGaussianProductCov(invSigmaHSq, invFootprintCov);
	Vector2f finalMu = getFlatGaussianProductMean(finalCov, invSigmaHSq, invFootprintCov, u0, footprintMean);
	Float finalC = getFlatGaussianProductScalingCoeff(finalMu, c1, c2, u0, footprintMean, invSigmaHSq, invFootprintCov);
    if (finalC < 0.f)
        printf("Less than 0 final C\n");
	// Integration over combined, final Gaussian
	return finalC * 2 * Pi * finalCov; // sqrt(norm of final cov matrix)
}
// Microfacet Utility Functions
static void BeckmannSample11(Float cosThetaI, Float U1, Float U2,
                             Float *slope_x, Float *slope_y) {
    /* Special case (normal incidence) */
    if (cosThetaI > .9999) {
        Float r = std::sqrt(-std::log(1.0f - U1));
        Float sinPhi = std::sin(2 * Pi * U2);
        Float cosPhi = std::cos(2 * Pi * U2);
        *slope_x = r * cosPhi;
        *slope_y = r * sinPhi;
        return;
    }

    /* The original inversion routine from the paper contained
       discontinuities, which causes issues for QMC integration
       and techniques like Kelemen-style MLT. The following code
       performs a numerical inversion with better behavior */
    Float sinThetaI =
        std::sqrt(std::max((Float)0, (Float)1 - cosThetaI * cosThetaI));
    Float tanThetaI = sinThetaI / cosThetaI;
    Float cotThetaI = 1 / tanThetaI;

    /* Search interval -- everything is parameterized
       in the Erf() domain */
    Float a = -1, c = Erf(cotThetaI);
    Float sample_x = std::max(U1, (Float)1e-6f);

    /* Start with a good initial guess */
    // Float b = (1-sample_x) * a + sample_x * c;

    /* We can do better (inverse of an approximation computed in
     * Mathematica) */
    Float thetaI = std::acos(cosThetaI);
    Float fit = 1 + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
    Float b = c - (1 + c) * std::pow(1 - sample_x, fit);

    /* Normalization factor for the CDF */
    static const Float SQRT_PI_INV = 1.f / std::sqrt(Pi);
    Float normalization =
        1 /
        (1 + c + SQRT_PI_INV * tanThetaI * std::exp(-cotThetaI * cotThetaI));

    int it = 0;
    while (++it < 10) {
        /* Bisection criterion -- the oddly-looking
           Boolean expression are intentional to check
           for NaNs at little additional cost */
        if (!(b >= a && b <= c)) b = 0.5f * (a + c);

        /* Evaluate the CDF and its derivative
           (i.e. the density function) */
        Float invErf = ErfInv(b);
        Float value =
            normalization *
                (1 + b + SQRT_PI_INV * tanThetaI * std::exp(-invErf * invErf)) -
            sample_x;
        Float derivative = normalization * (1 - invErf * tanThetaI);

        if (std::abs(value) < 1e-5f) break;

        /* Update bisection intervals */
        if (value > 0)
            c = b;
        else
            a = b;

        b -= value / derivative;
    }

    /* Now convert back into a slope value */
    *slope_x = ErfInv(b);

    /* Simulate Y component */
    *slope_y = ErfInv(2.0f * std::max(U2, (Float)1e-6f) - 1.0f);

    CHECK(!std::isinf(*slope_x));
    CHECK(!std::isnan(*slope_x));
    CHECK(!std::isinf(*slope_y));
    CHECK(!std::isnan(*slope_y));
}

static Vector3f BeckmannSample(const Vector3f &wi, Float alpha_x, Float alpha_y,
                               Float U1, Float U2) {
    // 1. stretch wi
    Vector3f wiStretched =
        Normalize(Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    Float slope_x, slope_y;
    BeckmannSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

    // 3. rotate
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return Normalize(Vector3f(-slope_x, -slope_y, 1.f));
}

// MicrofacetDistribution Method Definitions
MicrofacetDistribution::~MicrofacetDistribution() {}

Float BeckmannDistribution::D(const Vector3f &wh) const {
    Float tan2Theta = Tan2Theta(wh);
    if (std::isinf(tan2Theta)) return 0.;
    Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    return std::exp(-tan2Theta * (Cos2Phi(wh) / (alphax * alphax) +
                                  Sin2Phi(wh) / (alphay * alphay))) /
           (Pi * alphax * alphay * cos4Theta);
}

Float TrowbridgeReitzDistribution::D(const Vector3f &wh) const {
    Float tan2Theta = Tan2Theta(wh);
    if (std::isinf(tan2Theta)) return 0.;
    const Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    Float e =
        (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
        tan2Theta;
    return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
}

//TODO: Rewrite this
Float FlatGaussianElementsDistribution::D(const Vector3f &wh) const {
	Float sigmaR = 0.005f;
	Float h = 1.f / res.x;  // step size
	Float sigmaH = h / std::sqrt(8.f * std::log(2.f));  // std dev of Gaussian seeds
	Float invSigmaHSq = 1.f / (sigmaH * sigmaH);
	Float invSigmaRSq = 1.f / (sigmaR * sigmaR);

    int footprintSize = res.x / 8.f;

	Float footprintRadius = 0.5 * footprintSize;
	Float footprintVar = 0.5 * footprintRadius;    // distance between centers of footprints
	Float invCovFootprint = 1.f / (footprintVar * footprintVar);
	Vector2f uv = Vector2f(u, v);
    // printf("%f, %f (uv values)\n", u, v);
	
	// Sum over the relevant Gaussians
	// TODO: accelerate by calculating relevant bounds

    // Measure relevant contributions to u and v directions
	Vector2f localWh = Vector2f(Dot(wh, dpdu), Dot(wh, dpdv));
    printf("wh: %f, %f, %f\n", wh.x, wh.y, wh.z);

    printf("localWh: %f, %f\n", localWh.x, localWh.y);
    // if (localWh.Length() > 0.975) {
    //     return 0.f;
    // }
	// printf("at (%d, %d), normal: (%f, %f)\n", x, y, st.x, st.y);

	// printf("    summing values:\n");
    int halfFootprint = footprintSize * 0.5;
    int lowerX = Clamp(u*res.x - halfFootprint, 0, res.x - 1);
    int lowerY = Clamp(v*res.y - halfFootprint, 0, res.y - 1);
    int upperX = Clamp(u*res.x + halfFootprint, 0, res.x - 1);
    int upperY = Clamp(v*res.y + halfFootprint, 0, res.y - 1);

    Float sum = 0;
    // for (int idx = 0; idx < res.x*res.y; ++idx) {
    //     Float contribution = evaluateFlatPNDF(
    //         gaussians[idx].c,
    //         uv - gaussians[idx].u,
    //         localWh - gaussians[idx].n,
    //         invSigmaHSq,
    //         invSigmaRSq,
    //         uv - gaussians[idx].u /* footprint mean */,
    //         invCovFootprint
    //     );
    //     sum += contribution;
    // }
    for (int x = lowerX; x < upperX; ++x) {
        for (int y = lowerY; y < upperY; ++y) {
            // printf("Sampling for (%d, %d): \n", x, y);
            // printf("From (%d, %d) to (%d, %d)\n", lowerX, lowerY, upperX, upperY);

            int idx = y*res.x + x;
            Float contribution = evaluateFlatPNDF(
                gaussians[idx].c,
                uv - gaussians[idx].u,
                localWh - gaussians[idx].n,
                invSigmaHSq,
                invSigmaRSq,
                uv - gaussians[idx].u /* footprint mean */,
                invCovFootprint
            );
            sum += contribution;
        }
    }
    sum *= ((Float)footprintSize / res.x);

    if (sum <= 0.f)
        printf("sum: %f ZERO\n", sum);
    else
        printf("sum: %f\n", sum);


// TODO: additional scaling factor dependent on footprint?
    // sum = Clamp(sum, 1e-35f, 1.f);
	
	// printf("Sample %d finished!\n", sample);

	
	
	return sum;
}

Float BeckmannDistribution::Lambda(const Vector3f &w) const {
    Float absTanTheta = std::abs(TanTheta(w));
    if (std::isinf(absTanTheta)) return 0.;
    // Compute _alpha_ for direction _w_
    Float alpha =
        std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    Float a = 1 / (alpha * absTanTheta);
    if (a >= 1.6f) return 0;
    return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

Float TrowbridgeReitzDistribution::Lambda(const Vector3f &w) const {
    Float absTanTheta = std::abs(TanTheta(w));
    if (std::isinf(absTanTheta)) return 0.;
    // Compute _alpha_ for direction _w_
    Float alpha =
        std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
}

//TODO: Rewrite this
Float FlatGaussianElementsDistribution::Lambda(const Vector3f &w) const {
	Float absTanTheta = std::abs(TanTheta(w));
	if (std::isinf(absTanTheta)) return 0.;
	// Compute _alpha_ for direction _w_
	Float alpha =
		std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
	Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
	return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
}

std::string BeckmannDistribution::ToString() const {
    return StringPrintf("[ BeckmannDistribution alphax: %f alphay: %f ]",
                        alphax, alphay);
}

std::string TrowbridgeReitzDistribution::ToString() const {
    return StringPrintf("[ TrowbridgeReitzDistribution alphax: %f alphay: %f ]",
                        alphax, alphay);
}

std::string FlatGaussianElementsDistribution::ToString() const {
	return StringPrintf("[ FlatGaussianElementsDistribution alphax: %f alphay: %f ]",
		alphax, alphay);
}

Vector3f BeckmannDistribution::Sample_wh(const Vector3f &wo,
                                         const Point2f &u) const {
    if (!sampleVisibleArea) {
        // Sample full distribution of normals for Beckmann distribution

        // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
        Float tan2Theta, phi;
        if (alphax == alphay) {
            Float logSample = std::log(u[0]);
            if (std::isinf(logSample)) logSample = 0;
            tan2Theta = -alphax * alphax * logSample;
            phi = u[1] * 2 * Pi;
        } else {
            // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
            // distribution
            Float logSample = std::log(u[0]);
            if (std::isinf(logSample)) logSample = 0;
            phi = std::atan(alphay / alphax *
                            std::tan(2 * Pi * u[1] + 0.5f * Pi));
            if (u[1] > 0.5f) phi += Pi;
            Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
            Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            tan2Theta = -logSample /
                        (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
        }

        // Map sampled Beckmann angles to normal direction _wh_
        Float cosTheta = 1 / std::sqrt(1 + tan2Theta);
        Float sinTheta = std::sqrt(std::max((Float)0, 1 - cosTheta * cosTheta));
        Vector3f wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
        return wh;
    } else {
        // Sample visible area of normals for Beckmann distribution
        Vector3f wh;
        bool flip = wo.z < 0;
        wh = BeckmannSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if (flip) wh = -wh;
        return wh;
    }
}

static void TrowbridgeReitzSample11(Float cosTheta, Float U1, Float U2,
                                    Float *slope_x, Float *slope_y) {
    // special case (normal incidence)
    if (cosTheta > .9999) {
        Float r = sqrt(U1 / (1 - U1));
        Float phi = 6.28318530718 * U2;
        *slope_x = r * cos(phi);
        *slope_y = r * sin(phi);
        return;
    }

    Float sinTheta =
        std::sqrt(std::max((Float)0, (Float)1 - cosTheta * cosTheta));
    Float tanTheta = sinTheta / cosTheta;
    Float a = 1 / tanTheta;
    Float G1 = 2 / (1 + std::sqrt(1.f + 1.f / (a * a)));

    // sample slope_x
    Float A = 2 * U1 / G1 - 1;
    Float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10;
    Float B = tanTheta;
    Float D = std::sqrt(
        std::max(Float(B * B * tmp * tmp - (A * A - B * B) * tmp), Float(0)));
    Float slope_x_1 = B * tmp - D;
    Float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    // sample slope_y
    Float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    } else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    Float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * std::sqrt(1.f + *slope_x * *slope_x);

    CHECK(!std::isinf(*slope_y));
    CHECK(!std::isnan(*slope_y));
}

static Vector3f TrowbridgeReitzSample(const Vector3f &wi, Float alpha_x,
                                      Float alpha_y, Float U1, Float U2) {
    // 1. stretch wi
    Vector3f wiStretched =
        Normalize(Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    Float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

    // 3. rotate
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return Normalize(Vector3f(-slope_x, -slope_y, 1.));
}

Vector3f TrowbridgeReitzDistribution::Sample_wh(const Vector3f &wo,
                                                const Point2f &u) const {
    Vector3f wh;
    if (!sampleVisibleArea) {
        Float cosTheta = 0, phi = (2 * Pi) * u[1];
        if (alphax == alphay) {
            Float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
            cosTheta = 1 / std::sqrt(1 + tanTheta2);
        } else {
            phi =
                std::atan(alphay / alphax * std::tan(2 * Pi * u[1] + .5f * Pi));
            if (u[1] > .5f) phi += Pi;
            Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
            const Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            const Float alpha2 =
                1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
            cosTheta = 1 / std::sqrt(1 + tanTheta2);
        }
        Float sinTheta =
            std::sqrt(std::max((Float)0., (Float)1. - cosTheta * cosTheta));
        wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
    } else {
        bool flip = wo.z < 0;
        wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if (flip) wh = -wh;
    }
    return wh;
}

//TODO: Rewrite this
Vector3f FlatGaussianElementsDistribution::Sample_wh(const Vector3f &wo,
	const Point2f &u) const {

    // Vector3f wh;

    // Float sigmaR = 0.005f;
    // Float h = 1.f / res.x;  // step size
    // Float sigmaH = h / std::sqrt(8.f * std::log(2.f));  // std dev of Gaussian seeds
    // Float invSigmaHSq = 1.f / (sigmaH * sigmaH);
    // Float invSigmaRSq = 1.f / (sigmaR * sigmaR);

    // int footprintSize = res.x / 8.f;

    // Float footprintRadius = 0.5 * footprintSize;
    // Float footprintVar = 0.5 * footprintRadius;    // distance between centers of footprints
    // Float invCovFootprint = 1.f / (footprintVar * footprintVar);
    // Vector2f uv = Vector2f(u, v);
    // // printf("%f, %f (uv values)\n", u, v);
    
    // // Sum over the relevant Gaussians
    // // TODO: accelerate by calculating relevant bounds
    // Vector2f localWh = Vector2f(wh.x, wh.y);
    // // if (localWh.Length() > 0.975) {
    // //     return 0.f;
    // // }
    // // printf("at (%d, %d), normal: (%f, %f)\n", x, y, st.x, st.y);

    // // printf("    summing values:\n");
    // int halfFootprint = footprintSize * 0.5;
    // int lowerX = Clamp(u*res.x - halfFootprint, 0, res.x - 1);
    // int lowerY = Clamp(v*res.y - halfFootprint, 0, res.y - 1);
    // int upperX = Clamp(u*res.x + halfFootprint, 0, res.x - 1);
    // int upperY = Clamp(v*res.y + halfFootprint, 0, res.y - 1);

    // Vector2f uv = Vector2f(u.x, u.y);
    // Float curMinDistance = MaxFloat;

    // for (int idx = 0; idx < res.x*res.y; ++idx) {
    //     Float distance = (gaussians[idx].u - uv).LengthSquared();
    //     if (distance < curMinDistance) {
    //         curMinDistance = distance;
    //         wh = Vector3f(gaussians[idx].n.x, gaussians[idx].n.y, wo.z);
    //         // if (wo.z < 0)
    //         //     wh.z = -wo.z;
    //     }
    // }
    // return wh;


	Vector3f wh;
	if (!sampleVisibleArea) {
		Float cosTheta = 0, phi = (2 * Pi) * u[1];
		if (alphax == alphay) {
			Float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
			cosTheta = 1 / std::sqrt(1 + tanTheta2);
		}
		else {
			phi =
				std::atan(alphay / alphax * std::tan(2 * Pi * u[1] + .5f * Pi));
			if (u[1] > .5f) phi += Pi;
			Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
			const Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
			const Float alpha2 =
				1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
			Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
			cosTheta = 1 / std::sqrt(1 + tanTheta2);
		}
		Float sinTheta =
			std::sqrt(std::max((Float)0., (Float)1. - cosTheta * cosTheta));
		wh = SphericalDirection(sinTheta, cosTheta, phi);
		if (!SameHemisphere(wo, wh)) wh = -wh;
	}
	else {
		bool flip = wo.z < 0;
		wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
		if (flip) wh = -wh;
	}
    // printf("(%f, %f): uv\n", u.x, u.y);
	return wh;
}



Float MicrofacetDistribution::Pdf(const Vector3f &wo,
                                  const Vector3f &wh) const {
    if (sampleVisibleArea)
        return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
    else
        return D(wh) * AbsCosTheta(wh);
}



// FlatGaussianElementsDistribution::FlatGaussianElementsDistribution(Float alphax, Float alphay,
//                                     bool samplevis = true)
//     : MicrofacetDistribution(samplevis), alphax(alphax), alphay(alphay) {
// }

}  // namespace pbrt
