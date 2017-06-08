// Kevin Egan

#include <stdio.h>
#include <stdlib.h>

#include "pbrt.h"
#include "reflection.h"
#include "sampling.h"
#include "memory.h"
#include "api.h"
#include "paramset.h"
#include "fileutil.h"
#include "imageio.h"
#include "shapes/disk.h"

using namespace pbrt;

static MemoryArena arena;
static RNG rng;

// extract the red channel from a Spectrum class
double spectrumRedValue(const Spectrum& s) { return s[0]; }

typedef void (*CreateBSDFFunc)(BSDF* bsdf);

void createLambertian(BSDF* bsdf);
void createOrenNayar0(BSDF* bsdf);
void createOrenNayar20(BSDF* bsdf);
void createMicrofacet(BSDF* bsdf, bool beckmann, bool samplevisible,
                      float roughx, float roughy);
void createMicrofacet30and0(BSDF* bsdf);
void createFresnelBlend(BSDF* bsdf, bool beckmann, bool samplevisible,
                        float roughx, float roughy);

typedef void (*GenSampleFunc)(BSDF* bsdf, const Vector3f& wo, Vector3f* wi,
                              Float* pdf, Spectrum* f);

void Gen_Sample_f(BSDF* bsdf, const Vector3f& wo, Vector3f* wi, Float* pdf,
                  Spectrum* f);
void Gen_CosHemisphere(BSDF* bsdf, const Vector3f& wo, Vector3f* wi, Float* pdf,
                       Spectrum* f);
void Gen_UniformHemisphere(BSDF* bsdf, const Vector3f& wo, Vector3f* wi,
                           Float* pdf, Spectrum* f);

struct FlatlandGaussianElement {
    Float u;
    Float n;
};

Float flatlandNormal(Float x) {
    return std::sin(x*10.f) * 0.16f + std::cos(x*30.f) * 0.1f + 0.5;
}

int createFlatland(int argc, char *argv[]) {
    const char *outFilename = "flatlandFlatApproximation.exr";

    Float sigmaR = 0.01f;
    Point2i outputDim(256,256);
    int i;
    for (i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') break;
        if (!strcmp(argv[i], "--outputdim")) {
            ++i;
            outputDim.x = atoi(argv[i]);
            ++i;
            outputDim.y = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--sigmar")) {
            ++i;
            sigmaR = atof(argv[i]);
        }
    }

    std::unique_ptr<RGBSpectrum[]> outputImage(new RGBSpectrum[outputDim.x*outputDim.y]);

    // u v normal map
    // Flatland
    int m = 80; // number of Gaussians
    FlatlandGaussianElement* gaussians = new FlatlandGaussianElement[m];
    Float h = 1.f / m;  // step size
    Float sigmaH = h / std::sqrt(8.f * std::log(2.f));  // std dev of Gaussian seeds, as determined by paper

    for (int i = 0; i < m; ++i) {
        Float x = i*h;
        Float y = flatlandNormal(x);
        gaussians[i].u = x;
        gaussians[i].n = y;
    }

    // Row major order
    for (int y = 0; y < outputDim.y; ++y) {
        for (int x = 0; x < outputDim.x; ++x) {
            Float sum = 0;
            for (int i = 0; i < m; ++i) {
                Float u = ((Float)x) / outputDim.x;
                Float s = ((Float)y) / outputDim.y;
                Float duSquared = std::pow(u - gaussians[i].u, 2);
                
                Float positionBand = std::exp(-duSquared / (2 * sigmaH * sigmaH));
                Float dsSquared = std::pow(s - gaussians[i].n, 2);
                Float normalBand = std::exp(-dsSquared / (2 * sigmaR * sigmaR));

                sum += positionBand * normalBand;

            }
            outputImage.get()[y * outputDim.x + x] = RGBSpectrum(sum);
        }
    }

    WriteImage(outFilename, (Float *)outputImage.get(), Bounds2i(Point2i(0, 0), outputDim),
        outputDim);

    delete[] gaussians;

    return 0;
}

Vector2f generateUniformRandomNormal() {
    float u1 = rng.UniformFloat();
    float u2 = rng.UniformFloat();
    Vector3f hemisphereSample = UniformSampleHemisphere(Point2f(u1, u2));
    return Vector2f(hemisphereSample.x, hemisphereSample.y);
}

Vector2f generateSeedNormals(Vector2f dimensions) {
    // Sample for m*m Gaussian seeds
    // Float nSamples = m*m;
    // for (int i = 0; i < nSamples; ++i) {
    //     // uniform sampling for now. TODO: at least stratified? see sampling.h
    //     Float x = dimensions.x * rng.UniformFloat();
    //     Float y = dimensions.y * rng.UniformFloat();

    //     // TODO: sample from Beckmann distribution or Trowbridge
    //     gaussians[i].u = Vector2f(x, y);
    //     gaussians[i].n = generateUniformRandomNormal();
    //     // when integrating over all samples, we should get one
    //     gaussians[i].c = h*h * (1.f / 2*Pi) * invSigmaHSq * invSigmaRSq;
    // }
    return Vector2f(0,0);
}

struct FlatGaussianElement {
    Vector2f u;
    Vector2f n;

    Float c;

    // Float invCov[4][4];
};

Float getFlatGaussian2DConstant(Float c, Vector2f s, Float invSigmaRSq) {
    return c * std::exp(-0.5 * invSigmaRSq * s.LengthSquared());
}

Float evaluate2DFlatGaussian(Float c, Vector2f u, Vector2f u0, Float invCov) {
    return c * std::exp(-0.5 * (invCov * (u - u0).LengthSquared()));
}
Float evaluate4DFlatGaussian(Float c, Vector2f u, Vector2f s, Float invSigmaHSq, Float invSigmaRSq) {
    return c * std::exp(-0.5 * (invSigmaHSq * u.LengthSquared() + invSigmaRSq * s.LengthSquared()));
}

Float getFlatGaussianProductCov(Float invSigmaHSq, Float invFootprintCov) {
    return 1.f / (invSigmaHSq + invFootprintCov);
}

Vector2f getFlatGaussianProductMean(Float finalCov, Float invCov1, Float invCov2, Vector2f mu1, Vector2f mu2) {
    return finalCov * (invCov1 * mu1 + invCov2 * mu2);
}

Float getFlatGaussianProductScalingCoeff(Vector2f finalMu, Float c1, Float c2, Vector2f mu1, Vector2f mu2, 
                                        Float invCov1, Float invCov2) {
    return evaluate2DFlatGaussian(c1, finalMu, mu1, invCov1) * evaluate2DFlatGaussian(c2, finalMu, mu2, invCov2);
}

Float evaluateFlatPNDF(Float c, Vector2f u, Vector2f s, Float invSigmaHSq, Float invSigmaRSq,
                        Vector2f footprintMean, Float invFootprintCov) {

    Vector2f u0(0,0);   // Flat Gaussian has a diagonal matrix for invCov
    Float c1 = getFlatGaussian2DConstant(c, s, invSigmaRSq);
    Float c2 = 1.f / (2 * Pi * (1.f / invFootprintCov));    // normalizing the integration
    Float finalCov = getFlatGaussianProductCov(invSigmaHSq, invFootprintCov);
    Vector2f finalMu = getFlatGaussianProductMean(finalCov, invSigmaHSq, invFootprintCov, u0, footprintMean);
    Float finalC = getFlatGaussianProductScalingCoeff(finalMu, c1, c2, u0, footprintMean, invSigmaHSq, invFootprintCov);
    // Integration over combined, final Gaussian
    return finalC * 2 * Pi * finalCov; // sqrt(norm of final cov matrix)
}

Vector2f sampleNormalFromNormalMap(const RGBSpectrum* normalMap, int size, int x, int y) {
    // bilinear interpolation
    // Assuming square size
    x = Clamp(x, 0, size - 1);
    y = Clamp(y, 0, size - 1);
    RGBSpectrum rgb = normalMap[y*size + x];
    Float colors[3];
    rgb.ToRGB(colors);
    return Vector2f(colors[0] /* r */, colors[1] /* g */) * 2.f - Vector2f(1.f, 1.f);
}


Vector2f bilerpNormals(Float s, Float t, 
                        Vector2f v00, Vector2f v01,
                        Vector2f v10, Vector2f v11) {
    return (1.f - s)*(1.f - t)*v00 +
            (1.f - s)*t*v01 +
            s*(1.f - t)*v10 +
            s*t*v11;
}

Vector2f interpNormalFromNormalMap(const RGBSpectrum* normalMap, int size, Vector2f uv) {
    Vector2f st = uv*size;
    Vector2i xy = Vector2i(st.x, st.y); // truncated
    st = st - Vector2f(xy.x, xy.y);   // interp coordinates
    Vector2f v00 = sampleNormalFromNormalMap(normalMap, size, xy.x, xy.y);
    Vector2f v01 = sampleNormalFromNormalMap(normalMap, size, xy.x, xy.y + 1);
    Vector2f v10 = sampleNormalFromNormalMap(normalMap, size, xy.x + 1, xy.y);
    Vector2f v11 = sampleNormalFromNormalMap(normalMap, size, xy.x + 1, xy.y + 1);

    return bilerpNormals(st.x, st.y, v00, v01, v10, v11);
}

int create4DPNDF(int argc, char* argv[]) {
    const char *inFilename = "normals.png";
    const char *outFilename = "4DFlatApproximation.exr";

    Float sigmaR = 0.005f;
    Point2i outputDim(256,256);
    int i;
    for (i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') break;
        if (!strcmp(argv[i], "--outputdim")) {
            ++i;
            outputDim.x = atoi(argv[i]);
            ++i;
            outputDim.y = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--sigmar")) {
            ++i;
            sigmaR = atof(argv[i]);
        }
    }

    Point2i res;
    std::unique_ptr<RGBSpectrum[]> normalMapImage(ReadImage(inFilename, &res));
    if (!normalMapImage) {
        fprintf(stderr, "%s: unable to read image\n", inFilename);
        return 1;
    }

    std::unique_ptr<RGBSpectrum[]> outputImage(new RGBSpectrum[outputDim.x*outputDim.y]);

    // u v normal map
    // 4D Gaussians, Gaussians on each dimension
    int m = outputDim.x; // number of Gaussians per dimension, determined by texile size
    pbrt::FlatGaussianElement* gaussians = new pbrt::FlatGaussianElement[m*m];  // square distribution
    Float h = 1.f / m;  // step size
    Float sigmaH = h / std::sqrt(8.f * std::log(2.f));  // std dev of Gaussian seeds
    Float invSigmaHSq = 1.f / (sigmaH * sigmaH);
    Float invSigmaRSq = 1.f / (sigmaR * sigmaR);

    // Float invCov[4][4];
    // invCov[0][0] = 
    // Matrix4x4 invCov(invSigmaHSq, Float t01, Float t02, Float t03, 
    //                  Float t10, invSigmaHSq, Float t12, Float t13, 
    //                  Float t20, Float t21, invSigmaRSq, Float t23, 
    //                  Float t30, Float t31, Float t32, invSigmaRSq);

    // Sample for m*m normal map

    for (int y = 0; y < res.y; ++y) {
        for (int x = 0; x < res.x; ++x) {
            int idx = y*res.x + x;
            gaussians[idx].u = Vector2f(x * (1.f / outputDim.x), y * (1.f / outputDim.y));
            gaussians[idx].n = sampleNormalFromNormalMap(normalMapImage.get(), res.x, x, y);
            printf("at (%d, %d), normal: (%f, %f)\n", x, y, gaussians[idx].n.x, gaussians[idx].n.y);

            // when integrating over all samples, we should get one
            gaussians[idx].c = h*h / ((4*Pi*Pi) * (sigmaH*sigmaH) * (sigmaR*sigmaR));
        }
    }

    // For testing purposes. We'll generate 4 different incident light directions to sample the BRDF
    int nDirectionSamples = 4;

    // Footprint stats for this one footprint
    Float footprintRadius = 0.25;
    Float footprintVar = 0.5 * footprintRadius;    // distance between centers of footprints
    Float invCovFootprint = 1.f / (footprintVar * footprintVar);
    Vector2f footprintCenter = Vector2f(outputDim.x / 2.f, outputDim.y / 2.f);
    Vector2f footprintMean = Vector2f(footprintCenter.x / outputDim.x, footprintCenter.y / outputDim.y);
    Float maxContribution = 0.f;
    Float maxSum = 0.f;
    Vector2i maxSumPixel(0,0);
    Vector2f maxUV(0, 0);
    // Row major order
    for (int y = 0; y < outputDim.y; ++y) {
        for (int x = 0; x < outputDim.x; ++x) {
            Float sum = 0;
            // Sum over the relevant Gaussians
            // TODO: accelerate by calculating relevant bounds
            // printf("Sampling for (%d, %d): \n", x, y);
            for (int sample = 0; sample < nDirectionSamples; ++sample) {
                // Remapping the st space???
                Vector2f st = Vector2f((x + rng.UniformFloat()) * (1.f / outputDim.x), 
                    (y + rng.UniformFloat()) * (1.f / outputDim.y));
                st = st * 2.f - Vector2f(1.f, 1.f);
                // printf("at (%d, %d), normal: (%f, %f)\n", x, y, st.x, st.y);

                // Vector2f st = Vector2f(rng.UniformFloat(), rng.UniformFloat());
                Vector2f uv = Vector2f(((Float)(x)) / outputDim.x, ((Float)(y)) / outputDim.y);
                // printf("    summing values:\n");
                for (int idx = 0; idx < m*m; ++idx) {
                    Float contribution = evaluateFlatPNDF(
                        gaussians[idx].c, 
                        uv - gaussians[idx].u, 
                        st - gaussians[idx].n,
                        invSigmaHSq,
                        invSigmaRSq,
                        footprintMean - gaussians[idx].u,
                        invCovFootprint
                    );
                    sum += contribution;
                    if (contribution > maxContribution) {
                        maxContribution = contribution;
                        maxUV = uv;
                    }
                }
                // printf("Sample %d finished!\n", sample);
            }

            // TODO: additional scaling factor dependent on footprint?
            sum /= (Float)nDirectionSamples;
            if (sum > 0.f) {
                printf("Sampling for (%d, %d): ", x, y);
                printf("%f\n", sum);
                if (sum > maxSum) {
                    maxSum = sum;
                    maxSumPixel = Vector2i(x, y);
                }
            }
            outputImage.get()[y * outputDim.x + x] = RGBSpectrum(sum);
        }
        printf("Max: at uv(%f, %f): %f\n", maxUV.x, maxUV.y, maxContribution);
    }

    printf("OVERALL max: at uv(%f, %f): %f\n", maxUV.x, maxUV.y, maxContribution);
    printf("OVERALL MAX SUM: at xy(%d, %d): %f\n", maxSumPixel.x, maxSumPixel.y, maxSum);

    WriteImage(outFilename, (Float *)outputImage.get(), Bounds2i(Point2i(0, 0), outputDim),
        outputDim);

    delete[] gaussians;

    return 0;
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 1; // Warning and above.

    if (!strcmp(argv[1], "flatland"))
        return createFlatland(argc - 2, argv + 2);
    else if (!strcmp(argv[1], "4dflatpndf"))
        return create4DPNDF(argc-2, argv+2);

    Options opt;
    pbrtInit(opt);

    // number of monte carlo estimates
    // const int estimates = 1;
    const int estimates = 10000000;

    // radiance of uniform environment map
    const double environmentRadiance = 1.0;

    fprintf(stderr,
            "outgoing radiance from a surface viewed\n"
            "straight on with uniform lighting\n\n"
            "    uniform incoming radiance = %.3f\n"
            "    monte carlo samples = %d\n\n\n",
            environmentRadiance, estimates);

    CreateBSDFFunc BSDFFuncArray[] = {
        createLambertian,
        createOrenNayar0,
        createOrenNayar20,
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, true, 0.5, 0.5); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, true, 0.5, 0.5); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, true, 0.2, 0.1); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, true, 0.2, 0.1); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, true, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, true, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, true, 0.33, 0.033); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, true, 0.33, 0.033); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, false, 0.5, 0.5); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, false, 0.5, 0.5); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, false, 0.2, 0.1); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, false, 0.2, 0.1); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, false, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, false, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, true, false, 0.33, 0.033); },
        [](BSDF* bsdf) -> void
        { createMicrofacet(bsdf, false, false, 0.33, 0.033); },
        [](BSDF* bsdf) -> void
        { createFresnelBlend(bsdf, true, true, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createFresnelBlend(bsdf, false, true, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createFresnelBlend(bsdf, true, false, 0.15, 0.25); },
        [](BSDF* bsdf) -> void
        { createFresnelBlend(bsdf, false, false, 0.15, 0.25); },
    };

    const char* BSDFFuncDescripArray[] = {
        "Lambertian",
        "Oren Nayar (sigma 0)",
        "Oren Nayar (sigma 20)",
        "Beckmann (roughness 0.5, sample visible mf area)",
        "Trowbridge-Reitz (roughness 0.5, sample visible mf area)",
        "Beckmann (roughness 0.2/0.1, sample visible mf area)",
        "Trowbridge-Reitz (roughness 0.2/0.1, sample visible mf area)",
        "Beckmann (roughness 0.15/0.25, sample visible mf area)",
        "Trowbridge-Reitz (roughness 0.15/0.25, sample visible mf area)",
        "Beckmann (roughness 0.33/0.033, sample visible mf area)",
        "Trowbridge-Reitz (roughness 0.33/0.033, sample visible mf area)",
        "Beckmann (roughness 0.5, traditional sample wh)",
        "Trowbridge-Reitz (roughness 0.5, traditional sample wh)",
        "Beckmann (roughness 0.2/0.1, traditional sample wh)",
        "Trowbridge-Reitz (roughness 0.2/0.1, traditional sample wh)",
        "Beckmann (roughness 0.15/0.25, traditional sample wh)",
        "Trowbridge-Reitz (roughness 0.15/0.25, traditional sample wh)",
        "Beckmann (roughness 0.33/0.033, traditional sample wh)",
        "Trowbridge-Reitz (roughness 0.33/0.033, traditional sample wh)",
        "Fresnel Blend Beckmann (roughness 0.15/0.25, sample visible mf area)",
        "Fresnel Blend Trowbridge-Reitz (roughness 0.15/0.25, sample visible mf area)",
        "Fresnel Blend Beckmann (roughness 0.15/0.25, traditional sample wh)",
        "Fresnel Blend Trowbridge-Reitz (roughness 0.15/0.25, traditional sample wh)",
    };

    GenSampleFunc SampleFuncArray[] = {
        Gen_Sample_f,
        // CO        Gen_CosHemisphere,
        // CO        Gen_UniformHemisphere,
    };

    const char* SampleFuncDescripArray[] = {
        "BSDF Importance Sampling",
        // CO        "Cos Hemisphere",
        // CO        "Uniform Hemisphere",
    };

    int numModels = sizeof(BSDFFuncArray) / sizeof(BSDFFuncArray[0]);
    int numModelsDescrip =
        sizeof(BSDFFuncDescripArray) / sizeof(BSDFFuncDescripArray[0]);
    int numGenerators = sizeof(SampleFuncArray) / sizeof(SampleFuncArray[0]);
    int numGeneratorsDescrip =
        sizeof(SampleFuncDescripArray) / sizeof(SampleFuncDescripArray[0]);

    if (numModels != numModelsDescrip) {
        fprintf(stderr,
                "BSDFFuncArray and BSDFFuncDescripArray out of sync!\n");
        exit(1);
    }

    if (numGenerators != numGeneratorsDescrip) {
        fprintf(stderr,
                "SampleFuncArray and SampleFuncDescripArray out of sync!\n");
        exit(1);
    }

    // for each bsdf model
    for (int model = 0; model < numModels; model++) {
        BSDF* bsdf;

        // create BSDF which requires creating a Shape, casting a Ray
        // that hits the shape to get a SurfaceInteraction object.
        {
            Transform t = RotateX(-90);
            bool reverseOrientation = false;
            ParamSet p;

            std::shared_ptr<Shape> disk(
                new Disk(new Transform(t), new Transform(Inverse(t)),
                         reverseOrientation, 0., 1., 0, 360.));
            Point3f origin(
                0.1, 1, 0);  // offset slightly so we don't hit center of disk
            Vector3f direction(0, -1, 0);
            Float tHit;
            Ray r(origin, direction);
            SurfaceInteraction isect;
            disk->Intersect(r, &tHit, &isect);
            bsdf = ARENA_ALLOC(arena, BSDF)(isect);
            (BSDFFuncArray[model])(bsdf);
        }

        // facing directly at normal
        Vector3f woL = Normalize(Vector3f(0, 0, 1));
        Vector3f wo = bsdf->LocalToWorld(woL);
        // was bsdf->dgShading.nn
        const Normal3f n = Normal3f(bsdf->LocalToWorld(Vector3f(0, 0, 1)));

        // for each method of generating samples over the hemisphere
        for (int gen = 0; gen < numGenerators; gen++) {
            double redSum = 0.0;

            const int numHistoBins = 10;
            double histogram[numHistoBins][numHistoBins];
            for (int i = 0; i < numHistoBins; i++) {
                for (int j = 0; j < numHistoBins; j++) {
                    histogram[i][j] = 0;
                }
            }
            int badSamples = 0;
            int outsideSamples = 0;

            int warningTarget = 1;
            for (int sample = 0; sample < estimates; sample++) {
                Vector3f wi;
                Float pdf;
                Spectrum f;

                // sample hemisphere around bsdf, wo is fixed
                (SampleFuncArray[gen])(bsdf, wo, &wi, &pdf, &f);

                double redF = spectrumRedValue(f);

                // add hemisphere sample to histogram
                Vector3f wiL = bsdf->WorldToLocal(wi);
                float x = Clamp(wiL.x, -1.f, 1.f);
                float y = Clamp(wiL.y, -1.f, 1.f);
                float wiPhi = (atan2(y, x) + Pi) / (2.0 * Pi);
                float wiCosTheta = wiL.z;
                bool validSample = (wiCosTheta > 1e-7);
                if (wiPhi < -0.0001 || wiPhi > 1.0001 || wiCosTheta > 1.0001) {
                    // wiCosTheta can be less than 0
                    fprintf(stderr, "bad wi! %.3f %.3f %.3f, (%.3f %.3f)\n",
                            wiL[0], wiL[1], wiL[2], wiPhi, wiCosTheta);
                } else if (validSample) {
                    int histoPhi = (int)(wiPhi * numHistoBins);
                    if (histoPhi == numHistoBins)
                      --histoPhi;
                    int histoCosTheta = (int)(wiCosTheta * numHistoBins);
                    if (histoCosTheta == numHistoBins)
                      --histoCosTheta;
                    assert(histoPhi >= 0 && histoPhi < numHistoBins);
                    assert(histoCosTheta >= 0 && histoCosTheta < numHistoBins);
                    histogram[histoCosTheta][histoPhi] += 1.0 / pdf;
                }

                if (!validSample) {
                    outsideSamples++;
                } else if (pdf == 0.f || std::isnan(pdf) || redF < 0 ||
                           std::isnan(redF)) {
                    if (badSamples == warningTarget) {
                        fprintf(stderr,
                                "warning %d, bad sample %d! "
                                "pdf: %.3f, redF: %.3f\n",
                                warningTarget, sample, pdf, redF);
                        warningTarget *= 10;
                    }
                    badSamples++;
                } else {
                    // outgoing radiance estimate =
                    //   bsdf * incomingRadiance * cos(wi) / pdf
                    redSum += redF * environmentRadiance * AbsDot(wi, n) / pdf;
                }
            }
            int goodSamples = estimates - badSamples;

            // print results
            fprintf(stderr,
                    "*** BRDF: '%s', Samples: '%s'\n\n"
                    "wi histogram showing the relative weight in each bin\n"
                    "  all entries should be close to 2pi = %.5f:\n"
                    "  (%d bad samples, %d outside samples)\n\n"
                    "                          phi bins\n",
                    BSDFFuncDescripArray[model], SampleFuncDescripArray[gen],
                    Pi * 2.0, badSamples, outsideSamples);
            double totalSum = 0.0;
            for (int i = 0; i < numHistoBins; i++) {
                fprintf(stderr, "  cos(theta) bin %02d:", i);
                for (int j = 0; j < numHistoBins; j++) {
                    fprintf(stderr, " %5.2f", histogram[i][j] * numHistoBins *
                                                  numHistoBins / goodSamples);
                    totalSum += histogram[i][j];
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr,
                    "\n  final average :  %.5f (error %.5f)\n\n"
                    "  radiance = %.5f\n\n",
                    totalSum / goodSamples, totalSum / goodSamples - Pi * 2.0,
                    redSum / goodSamples);
        }
    }

    pbrtCleanup();
    return 0;
}

void Gen_Sample_f(BSDF* bsdf, const Vector3f& wo, Vector3f* wi, Float* pdf,
                  Spectrum* f) {
    // only glossy or diffuse reflections (no specular reflections)
    BxDFType inflags = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE | BSDF_GLOSSY);
    BxDFType outflags;
    Point2f sample {rng.UniformFloat(), rng.UniformFloat()};
    *f = bsdf->Sample_f(wo, wi, sample, pdf, inflags, &outflags);

    // double check bsdf->Pdf() gives us the same answer
    Vector3f wiL = bsdf->WorldToLocal(*wi);
    float wiCosTheta = wiL.z;
    bool validSample = (wiCosTheta > 1e-7);

    if (validSample) {
        float verifyPdf = bsdf->Pdf(wo, *wi, inflags);
        float relErr = std::abs(verifyPdf - *pdf) / *pdf;
        if (relErr > 1e-3) {
            fprintf(stderr,
                    "BSDF::Pdf() doesn't match BSDF::Sample_f() !\n"
                    "  Sample_f pdf %.3f, Pdf pdf %.3f (rel error %f)\n"
                    "  wo %.3f %.3f %.3f, wi %.3f %.3f %.3f\n",
                    *pdf, verifyPdf, relErr, wo[0], wo[1], wo[2], (*wi)[0],
                    (*wi)[1], (*wi)[2]);
            fprintf(
                stderr,
                "blah! validSample %d, wiCosTheta %.3f, wiL %.3f %.3f %.3f\n",
                validSample, wiCosTheta, wiL[0], wiL[1], wiL[2]);
        }
    }
}

void Gen_CosHemisphere(BSDF* bsdf, const Vector3f& wo, Vector3f* wi, Float* pdf,
                       Spectrum* f) {
    float u1 = rng.UniformFloat();
    float u2 = rng.UniformFloat();
    Vector3f wiL = CosineSampleHemisphere(Point2f(u1, u2));
    *wi = bsdf->LocalToWorld(wiL);
    float cosTheta = wiL.z;
    *pdf = CosineHemispherePdf(cosTheta);

    *f = bsdf->f(wo, *wi);
}

void Gen_UniformHemisphere(BSDF* bsdf, const Vector3f& wo, Vector3f* wi,
                           Float* pdf, Spectrum* f) {
    float u1 = rng.UniformFloat();
    float u2 = rng.UniformFloat();
    Vector3f wiL = UniformSampleHemisphere(Point2f(u1, u2));
    *wi = bsdf->LocalToWorld(wiL);
    *pdf = UniformHemispherePdf();

    *f = bsdf->f(wo, *wi);
}

void createLambertian(BSDF* bsdf) {
    Spectrum Kd(1);
    bsdf->Add(ARENA_ALLOC(arena, LambertianReflection)(Kd));
}

void createMicrofacet(BSDF* bsdf, bool beckmann, bool samplevisible,
                      float roughx, float roughy) {
    Spectrum Ks(1);
    MicrofacetDistribution *distrib;
    if (beckmann) {
      Float alphax = BeckmannDistribution::RoughnessToAlpha(roughx);
      Float alphay = BeckmannDistribution::RoughnessToAlpha(roughy);
      distrib = ARENA_ALLOC(arena, BeckmannDistribution)(alphax, alphay,
                                                        samplevisible);
    }
    else {
      Float alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(roughx);
      Float alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(roughy);
      distrib = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(alphax, alphay,
                                                               samplevisible);
    }
    Fresnel* fresnel = ARENA_ALLOC(arena, FresnelNoOp)();
    BxDF* bxdf = ARENA_ALLOC(arena, MicrofacetReflection)(Ks, distrib, fresnel);
    bsdf->Add(bxdf);
}

void createFresnelBlend(BSDF* bsdf, bool beckmann, bool samplevisible,
                        float roughx, float roughy)
{
    Spectrum d(0.5);
    Spectrum s(0.5);
    MicrofacetDistribution *distrib;
    if (beckmann) {
      Float alphax = BeckmannDistribution::RoughnessToAlpha(roughx);
      Float alphay = BeckmannDistribution::RoughnessToAlpha(roughy);
      distrib = ARENA_ALLOC(arena, BeckmannDistribution)(alphax, alphay,
                                                        samplevisible);
    }
    else {
      Float alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(roughx);
      Float alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(roughy);
      distrib = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(alphax, alphay,
                                                               samplevisible);
    }
    BxDF* bxdf = ARENA_ALLOC(arena, FresnelBlend)(d, s, distrib);
    bsdf->Add(bxdf);
}

void createMicrofacet30and0(BSDF* bsdf, bool beckmann) {
    Spectrum Ks(0.5);
    MicrofacetDistribution *distrib1, *distrib2;
    if (beckmann) {
      Float alphax = BeckmannDistribution::RoughnessToAlpha(0.8);
      Float alphay = BeckmannDistribution::RoughnessToAlpha(0.8);
      distrib1 = ARENA_ALLOC(arena, BeckmannDistribution)(alphax, alphay);

      alphax = BeckmannDistribution::RoughnessToAlpha(0.01);
      alphay = BeckmannDistribution::RoughnessToAlpha(0.01);
      distrib2 = ARENA_ALLOC(arena, BeckmannDistribution)(alphax, alphay);
    }
    else {
      Float alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(0.8);
      Float alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(0.8);
      distrib1 = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(alphax, alphay);

      alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(0.01);
      alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(0.01);
      distrib2 = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(alphax, alphay);
    }

    Fresnel* fresnel = ARENA_ALLOC(arena, FresnelNoOp)();
    BxDF* bxdf1 =
        ARENA_ALLOC(arena, MicrofacetReflection)(Ks, distrib1, fresnel);
    bsdf->Add(bxdf1);
    BxDF* bxdf2 =
        ARENA_ALLOC(arena, MicrofacetReflection)(Ks, distrib2, fresnel);
    bsdf->Add(bxdf2);
}

void createOrenNayar0(BSDF* bsdf) {
    Spectrum Kd(1);
    float sigma = 0.0;
    BxDF* bxdf = ARENA_ALLOC(arena, OrenNayar)(Kd, sigma);
    bsdf->Add(bxdf);
}

void createOrenNayar20(BSDF* bsdf) {
    Spectrum Kd(1);
    float sigma = 20.0;
    BxDF* bxdf = ARENA_ALLOC(arena, OrenNayar)(Kd, sigma);
    bsdf->Add(bxdf);
}


