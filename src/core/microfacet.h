
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_MICROFACET_H
#define PBRT_CORE_MICROFACET_H

// core/microfacet.h*
#include "pbrt.h"
#include "geometry.h"


namespace pbrt {

	struct FlatGaussianElement {
		Vector2f u;
		Vector2f n;
		Float c;

		//  Float invCov[4][4];
	};

// MicrofacetDistribution Declarations
class MicrofacetDistribution {
  public:
    // MicrofacetDistribution Public Methods
    virtual ~MicrofacetDistribution();
    virtual Float D(const Vector3f &wh) const = 0;
    virtual Float Lambda(const Vector3f &w) const = 0;
    Float G1(const Vector3f &w) const {
        //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
        return 1 / (1 + Lambda(w));
    }
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    virtual Vector3f Sample_wh(const Vector3f &wo, const Point2f &u) const = 0;
    Float Pdf(const Vector3f &wo, const Vector3f &wh) const;
    virtual std::string ToString() const = 0;

  protected:
    // MicrofacetDistribution Protected Methods
    MicrofacetDistribution(bool sampleVisibleArea)
        : sampleVisibleArea(sampleVisibleArea) {}

    // MicrofacetDistribution Protected Data
    const bool sampleVisibleArea;
};

inline std::ostream &operator<<(std::ostream &os,
                                const MicrofacetDistribution &md) {
    os << md.ToString();
    return os;
}

class BeckmannDistribution : public MicrofacetDistribution {
  public:
    // BeckmannDistribution Public Methods
    static Float RoughnessToAlpha(Float roughness) {
        roughness = std::max(roughness, (Float)1e-3);
        Float x = std::log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x +
               0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }
    BeckmannDistribution(Float alphax, Float alphay, bool samplevis = true)
        : MicrofacetDistribution(samplevis), alphax(alphax), alphay(alphay) {}
    Float D(const Vector3f &wh) const;
    Vector3f Sample_wh(const Vector3f &wo, const Point2f &u) const;
    std::string ToString() const;

  private:
    // BeckmannDistribution Private Methods
    Float Lambda(const Vector3f &w) const;

    // BeckmannDistribution Private Data
    const Float alphax, alphay;
};

class TrowbridgeReitzDistribution : public MicrofacetDistribution {
  public:
    // TrowbridgeReitzDistribution Public Methods
    static inline Float RoughnessToAlpha(Float roughness);
    TrowbridgeReitzDistribution(Float alphax, Float alphay,
                                bool samplevis = true)
        : MicrofacetDistribution(samplevis), alphax(alphax), alphay(alphay) {}
    Float D(const Vector3f &wh) const;
    Vector3f Sample_wh(const Vector3f &wo, const Point2f &u) const;
    std::string ToString() const;

  private:
    // TrowbridgeReitzDistribution Private Methods
    Float Lambda(const Vector3f &w) const;

    // TrowbridgeReitzDistribution Private Data
    const Float alphax, alphay;
};

// MicrofacetDistribution Inline Methods
inline Float TrowbridgeReitzDistribution::RoughnessToAlpha(Float roughness) {
    roughness = std::max(roughness, (Float)1e-3);
    Float x = std::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
}

class FlatGaussianElementsDistribution : public MicrofacetDistribution {
     public:
         // FlatGaussianElementsDistribution Public Methods
         static inline Float RoughnessToAlpha(Float roughness);
         FlatGaussianElementsDistribution(Float alphax, Float alphay, 
                                    Float u, Float v, Vector3f dpdu, Vector3f dpdv, 
                                    Normal3f n,
                                    FlatGaussianElement* gaussians, Point2i res,
                                     bool samplevis = false)
			 : MicrofacetDistribution(samplevis), 
         dpdu(dpdu), dpdv(dpdv), n(n), alphax(alphax), alphay(alphay) {
			 this->u = u;
			 this->v = v;
			 this->gaussians = gaussians;
			 this->res = res;
		 }
         Float D(const Vector3f &wh) const;
         Vector3f Sample_wh(const Vector3f &wo, const Point2f &u) const;
		 Float getFlatGaussian2DConstant(Float c, Vector2f s, Float invSigmaRSq) const;
		 Float evaluate2DFlatGaussian(Float c, Vector2f u, Vector2f u0, Float invCov) const;
		 Float getFlatGaussianProductCov(Float invSigmaHSq, Float invFootprintCov) const;
		 Vector2f getFlatGaussianProductMean(Float finalCov, Float invCov1, Float invCov2, Vector2f mu1, Vector2f mu2) const;
		 Float getFlatGaussianProductScalingCoeff(Vector2f finalMu, Float c1, Float c2, Vector2f mu1, Vector2f mu2, Float invCov1, Float invCov2) const;
		 Float evaluateFlatPNDF(Float c, Vector2f u, Vector2f s, Float invSigmaHSq, Float invSigmaRSq, Vector2f footprintMean, Float invFootprintCov) const;
		 std::string ToString() const;

     private:
         // FlatGaussianElementsDistribution Private Methods
         Float Lambda(const Vector3f &w) const;


         // FlatGaussianElementsDistribution Private Data
         const Float alphax, alphay;

		 FlatGaussianElement* gaussians;
		 Float u;
		 Float v;
     Vector3f dpdu;
     Vector3f dpdv;
     Normal3f n;
		 Point2i res;
};

}  // namespace pbrt

#endif  // PBRT_CORE_MICROFACET_H
