
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


// materials/snow.cpp*
#include "materials/snow.h"
#include "spectrum.h"
#include "reflection.h"
#include "paramset.h"
#include "texture.h"
#include "interaction.h"
#include "imageio.h"

namespace pbrt {



// SnowMaterial Method Definitions
void SnowMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                               MemoryArena &arena,
                                               TransportMode mode,
                                               bool allowMultipleLobes) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(bumpMap, si);

    // Initialize BSDF for _SnowMaterial_
    Spectrum R = Kr->Evaluate(*si).Clamp();
    Spectrum T = Kt->Evaluate(*si).Clamp();
    Float urough = uRoughness->Evaluate(*si);
    Float vrough = vRoughness->Evaluate(*si);

    // Initialize _bsdf_ for smooth or rough dielectric
    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si, eta);

    if (R.IsBlack() && T.IsBlack()) return;

    bool isSpecular = urough == 0 && vrough == 0;
    if (isSpecular && allowMultipleLobes) {
        si->bsdf->Add(
            ARENA_ALLOC(arena, FresnelSpecular)(R, T, 1.f, eta, mode));
    } else {
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        MicrofacetDistribution *trDistrib =
            isSpecular ? nullptr
                       : ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(
                             urough, vrough);

        MicrofacetDistribution *distrib =
            isSpecular ? nullptr
                       : ARENA_ALLOC(arena, FlatGaussianElementsDistribution)(
                             urough, vrough, si->uv[0], si->uv[1], si->dpdu, si->dpdv, si->shading.n, 
                             gaussians, normalRes);
        if (!R.IsBlack()) {
            Fresnel *fresnel = ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta);
            if (isSpecular)
                si->bsdf->Add(
                    ARENA_ALLOC(arena, SpecularReflection)(R, fresnel));
            else
                si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
                    R, distrib, fresnel));
        }
        if (!T.IsBlack()) {
            if (isSpecular)
                si->bsdf->Add(ARENA_ALLOC(arena, SpecularTransmission)(
                    T, 1.f, eta, mode));
            else
                si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetTransmission)(
                    T, trDistrib, 1.f, eta, mode));
        }
    }

    Spectrum sig_a = scale * sigma_a->Evaluate(*si).Clamp();
    Spectrum sig_s = scale * sigma_s->Evaluate(*si).Clamp();
    si->bssrdf = ARENA_ALLOC(arena, TabulatedBSSRDF)(*si, this, mode, eta,
                                                     sig_a, sig_s, table);
}

FlatGaussianElement* SnowMaterial::ComputeGaussianMixture()
{
	const char *inFilename = "normals.png";

	Float sigmaR = 0.005f;

	Point2i res;
	std::unique_ptr<RGBSpectrum[]> normalMapImage(ReadImage(inFilename, &res));
	normalRes = res;
	if (!normalMapImage) {
		fprintf(stderr, "%s: unable to read image\n", inFilename);
		return nullptr;
	}
	Point2i outputDim(res);

	// u v normal map
	// 4D Gaussians, Gaussians on each dimension
	int m = outputDim.x; // number of Gaussians per dimension, determined by texile size
	FlatGaussianElement* gaussians = new FlatGaussianElement[m*m];  // square distribution
	Float h = 1.f / m;  // step size
	Float sigmaH = h / std::sqrt(8.f * std::log(2.f));  // std dev of Gaussian seeds
	Float invSigmaHSq = 1.f / (sigmaH * sigmaH);
	Float invSigmaRSq = 1.f / (sigmaR * sigmaR);

	// Sample for m*m normal map

	for (int y = 0; y < res.y; ++y) {
		for (int x = 0; x < res.x; ++x) {
			int idx = y*res.x + x;
			gaussians[idx].u = Vector2f(x * (1.f / outputDim.x), y * (1.f / outputDim.y));
			gaussians[idx].n = sampleNormalFromNormalMap(normalMapImage.get(), res.x, x, y);
		//	printf("at (%d, %d), normal: (%f, %f)\n", x, y, gaussians[idx].n.x, gaussians[idx].n.y);

			// when integrating over all samples, we should get one
			gaussians[idx].c = h*h / ((4 * Pi*Pi) * (sigmaH*sigmaH) * (sigmaR*sigmaR));
		}
	}
	return gaussians;

}

Vector2f SnowMaterial::sampleNormalFromNormalMap(const RGBSpectrum* normalMap, int size, int x, int y) {
	// bilinear interpolation
	// Assuming square size
	x = Clamp(x, 0, size - 1);
	y = Clamp(y, 0, size - 1);
	RGBSpectrum rgb = normalMap[y*size + x];
	Float colors[3];
	rgb.ToRGB(colors);
	return Vector2f(colors[0] /* r */, colors[1] /* g */) * 2.f - Vector2f(1.f, 1.f);
}

SnowMaterial *CreateSnowMaterial(const TextureParams &mp) {
    // g should be 0.93?
    // extinction coefficient is 1.38121547
    Float sig_a_rgb[3] = {0.153468386, 0.072695551, 0.001f},
        sig_s_rgb[3] = {1.227747084, 1.308519919, 1.38121547};

    Spectrum sig_a = Spectrum::FromRGB(sig_a_rgb),
             sig_s = Spectrum::FromRGB(sig_s_rgb);

    Float g = mp.FindFloat("g", -0.93f);

    printf("Spectrum %f\n", sig_a[0]);

    Float scale = mp.FindFloat("scale", 1.f);
    Float eta = mp.FindFloat("eta", 1.33f);

    std::shared_ptr<Texture<Spectrum>> sigma_a, sigma_s;
    sigma_a = mp.GetSpectrumTexture("sigma_a", sig_a);
    sigma_s = mp.GetSpectrumTexture("sigma_s", sig_s);

    std::shared_ptr<Texture<Spectrum>> Kr =
        mp.GetSpectrumTexture("Kr", Spectrum(1.f));
    std::shared_ptr<Texture<Spectrum>> Kt = mp.GetSpectrumTexture("Kt", Spectrum(1.f));

    std::shared_ptr<Texture<Float>> roughu =
        mp.GetFloatTexture("uroughness", 0.f);
    std::shared_ptr<Texture<Float>> roughv =
        mp.GetFloatTexture("vroughness", 0.f);
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");

    bool remapRoughness = mp.FindBool("remaproughness", true);
    return new SnowMaterial(scale, Kr, Kt, sigma_a, sigma_s, g, eta,
                                  roughu, roughv, bumpMap, remapRoughness);
}

}  // namespace pbrt
