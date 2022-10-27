#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#define OMP_THREADS 2
#define OPENCV_THREADS 4

#define TEST_PIXEL 0.2807

constexpr float GaussDetCoeff { 248.050213442f }; // 2*pi^3


/* All GMM params copied from https://www.hpl.hp.com/techreports/Compaq-DEC/CRL-98-11.pdf */
// NOTE: All values represented in RGB format

static constexpr float Skin_Mus[16][3] = {
    {73.53, 29.94, 17.76}, {249.71, 233.94, 217.49}, {161.68, 116.25, 96.95}, {186.07, 136.62, 114.40},
    {189.26, 98.37, 51.18}, {247.00, 152.20, 90.84}, {150.10, 72.66, 37.76}, {206.85, 171.09, 156.34},
    {212.78, 152.82, 120.04}, {234.87, 175.43, 138.94}, {151.19, 97.74, 74.59}, {120.52, 77.55, 59.82},
    {192.20, 119.62, 82.32}, {214.29, 136.08, 87.24}, {99.57, 54.33, 38.06}, {238.88, 203.08, 176.91}
};
static constexpr float Skin_Sigmas[16][3] = {
    {765.40, 121.44, 112.80}, {39.94, 154.44, 396.05}, {291.03, 60.48, 162.85}, {274.95, 64.60, 198.27},
    {633.18, 222.40, 250.69}, {65.23, 691.53, 609.92}, {408.63, 200.77, 257.57}, {530.08, 155.08, 572.79},
    {160.57, 84.52, 243.90}, {163.80, 121.57, 279.22}, {425.40, 73.56, 175.11}, {330.45, 70.34, 151.82},
    {152.76, 92.14, 259.15}, {204.90, 140.17, 270.19}, {448.13, 90.18, 151.29}, {178.38, 156.27, 404.99}
};
static constexpr float Skin_Ws[16] = {
    0.0294, 0.0331, 0.0654, 0.0756,
    0.0554, 0.0314, 0.0454, 0.0469, 
    0.0956, 0.0763, 0.1100, 0.0676, 
    0.0755, 0.0500, 0.0667, 0.0749
};
static constexpr float Nonskin_Mus[16][3] = {
    {254.37, 254.41, 253.82}, {9.39, 8.09, 8.52}, {96.57, 96.95, 91.53}, {160.44, 162.49, 159.06},
    {74.98, 63.23, 46.33}, {121.83, 60.88, 18.31}, {202.18, 154.88, 91.04}, {193.06, 201.93, 206.55},
    {51.88, 57.14, 61.55}, {30.88, 26.84, 25.32}, {44.97, 85.96, 131.95}, {236.02, 236.27, 230.70},
    {207.86, 191.20, 164.12}, {99.83, 148.11, 188.17}, {135.06, 131.92, 123.10}, {135.96, 103.89, 66.88}
};
static constexpr float Nonskin_Sigmas[16][3] = {
    {2.77, 2.81, 5.46}, {46.84, 33.59, 32.48}, {280.69, 156.79, 436.58}, {355.98, 115.89, 591.24},
    {414.84, 245.95, 361.27}, {2502.24, 1383.53, 237.18}, {957.42, 1766.94, 1582.52}, {562.88, 190.23, 447.28},
    {344.11, 191.77, 433.40}, {222.07, 118.65, 182.41}, {651.32, 840.52, 963.67}, {225.03, 117.29, 331.95},
    {494.04, 237.69, 533.52}, {955.88, 654.95, 916.70}, {350.35, 130.30, 388.43}, {806.44, 642.20, 350.36}
};
static constexpr float Nonskin_Ws[16] = {
    0.0637, 0.0516, 0.0864, 0.0636,
    0.0747, 0.0365, 0.0349, 0.0649, 
    0.0656, 0.1189, 0.0362, 0.0849, 
    0.0368, 0.0389, 0.0943, 0.0477
};

static constexpr float SkinPrior { 30.0f/100.0f };
static constexpr float NonskinPrior { 70.0f/100.0f };

static constexpr float PrecomputedSkin_Precisions[16][3] = {
    {1.0f / Skin_Sigmas[0][0], 1.0f / Skin_Sigmas[0][1], 1.0f / Skin_Sigmas[0][2]}, 
    {1.0f / Skin_Sigmas[1][0], 1.0f / Skin_Sigmas[1][1], 1.0f / Skin_Sigmas[1][2]}, 
    {1.0f / Skin_Sigmas[2][0], 1.0f / Skin_Sigmas[2][1], 1.0f / Skin_Sigmas[2][2]}, 
    {1.0f / Skin_Sigmas[3][0], 1.0f / Skin_Sigmas[3][1], 1.0f / Skin_Sigmas[3][2]}, 
    {1.0f / Skin_Sigmas[4][0], 1.0f / Skin_Sigmas[4][1], 1.0f / Skin_Sigmas[4][2]}, 
    {1.0f / Skin_Sigmas[5][0], 1.0f / Skin_Sigmas[5][1], 1.0f / Skin_Sigmas[5][2]}, 
    {1.0f / Skin_Sigmas[6][0], 1.0f / Skin_Sigmas[6][1], 1.0f / Skin_Sigmas[6][2]}, 
    {1.0f / Skin_Sigmas[7][0], 1.0f / Skin_Sigmas[7][1], 1.0f / Skin_Sigmas[7][2]}, 
    {1.0f / Skin_Sigmas[8][0], 1.0f / Skin_Sigmas[8][1], 1.0f / Skin_Sigmas[8][2]}, 
    {1.0f / Skin_Sigmas[9][0], 1.0f / Skin_Sigmas[9][1], 1.0f / Skin_Sigmas[9][2]}, 
    {1.0f / Skin_Sigmas[10][0], 1.0f / Skin_Sigmas[10][1], 1.0f / Skin_Sigmas[10][2]}, 
    {1.0f / Skin_Sigmas[11][0], 1.0f / Skin_Sigmas[11][1], 1.0f / Skin_Sigmas[11][2]}, 
    {1.0f / Skin_Sigmas[12][0], 1.0f / Skin_Sigmas[12][1], 1.0f / Skin_Sigmas[12][2]}, 
    {1.0f / Skin_Sigmas[13][0], 1.0f / Skin_Sigmas[13][1], 1.0f / Skin_Sigmas[13][2]}, 
    {1.0f / Skin_Sigmas[14][0], 1.0f / Skin_Sigmas[14][1], 1.0f / Skin_Sigmas[14][2]}, 
    {1.0f / Skin_Sigmas[15][0], 1.0f / Skin_Sigmas[15][1], 1.0f / Skin_Sigmas[15][2]}, 
};

static constexpr float PrecomputedNonskin_Precisions[16][3] = {
    {1.0f / Nonskin_Sigmas[0][0], 1.0f / Nonskin_Sigmas[0][1], 1.0f / Nonskin_Sigmas[0][2]}, 
    {1.0f / Nonskin_Sigmas[1][0], 1.0f / Nonskin_Sigmas[1][1], 1.0f / Nonskin_Sigmas[1][2]}, 
    {1.0f / Nonskin_Sigmas[2][0], 1.0f / Nonskin_Sigmas[2][1], 1.0f / Nonskin_Sigmas[2][2]}, 
    {1.0f / Nonskin_Sigmas[3][0], 1.0f / Nonskin_Sigmas[3][1], 1.0f / Nonskin_Sigmas[3][2]}, 
    {1.0f / Nonskin_Sigmas[4][0], 1.0f / Nonskin_Sigmas[4][1], 1.0f / Nonskin_Sigmas[4][2]}, 
    {1.0f / Nonskin_Sigmas[5][0], 1.0f / Nonskin_Sigmas[5][1], 1.0f / Nonskin_Sigmas[5][2]}, 
    {1.0f / Nonskin_Sigmas[6][0], 1.0f / Nonskin_Sigmas[6][1], 1.0f / Nonskin_Sigmas[6][2]}, 
    {1.0f / Nonskin_Sigmas[7][0], 1.0f / Nonskin_Sigmas[7][1], 1.0f / Nonskin_Sigmas[7][2]}, 
    {1.0f / Nonskin_Sigmas[8][0], 1.0f / Nonskin_Sigmas[8][1], 1.0f / Nonskin_Sigmas[8][2]}, 
    {1.0f / Nonskin_Sigmas[9][0], 1.0f / Nonskin_Sigmas[9][1], 1.0f / Nonskin_Sigmas[9][2]}, 
    {1.0f / Nonskin_Sigmas[10][0], 1.0f / Nonskin_Sigmas[10][1], 1.0f / Nonskin_Sigmas[10][2]}, 
    {1.0f / Nonskin_Sigmas[11][0], 1.0f / Nonskin_Sigmas[11][1], 1.0f / Nonskin_Sigmas[11][2]}, 
    {1.0f / Nonskin_Sigmas[12][0], 1.0f / Nonskin_Sigmas[12][1], 1.0f / Nonskin_Sigmas[12][2]}, 
    {1.0f / Nonskin_Sigmas[13][0], 1.0f / Nonskin_Sigmas[13][1], 1.0f / Nonskin_Sigmas[13][2]}, 
    {1.0f / Nonskin_Sigmas[14][0], 1.0f / Nonskin_Sigmas[14][1], 1.0f / Nonskin_Sigmas[14][2]}, 
    {1.0f / Nonskin_Sigmas[15][0], 1.0f / Nonskin_Sigmas[15][1], 1.0f / Nonskin_Sigmas[15][2]}, 
};

static constexpr float PrecomputedSkin_Sigmas_det[16] = {
    Skin_Sigmas[0][0] * Skin_Sigmas[0][1] * Skin_Sigmas[0][2],
    Skin_Sigmas[1][0] * Skin_Sigmas[1][1] * Skin_Sigmas[1][2],
    Skin_Sigmas[2][0] * Skin_Sigmas[2][1] * Skin_Sigmas[2][2],
    Skin_Sigmas[3][0] * Skin_Sigmas[3][1] * Skin_Sigmas[3][2],
    Skin_Sigmas[4][0] * Skin_Sigmas[4][1] * Skin_Sigmas[4][2],
    Skin_Sigmas[5][0] * Skin_Sigmas[5][1] * Skin_Sigmas[5][2],
    Skin_Sigmas[6][0] * Skin_Sigmas[6][1] * Skin_Sigmas[6][2],
    Skin_Sigmas[7][0] * Skin_Sigmas[7][1] * Skin_Sigmas[7][2],
    Skin_Sigmas[8][0] * Skin_Sigmas[8][1] * Skin_Sigmas[8][2],
    Skin_Sigmas[9][0] * Skin_Sigmas[9][1] * Skin_Sigmas[9][2],
    Skin_Sigmas[10][0] * Skin_Sigmas[10][1] * Skin_Sigmas[10][2],
    Skin_Sigmas[11][0] * Skin_Sigmas[11][1] * Skin_Sigmas[11][2],
    Skin_Sigmas[12][0] * Skin_Sigmas[12][1] * Skin_Sigmas[12][2],
    Skin_Sigmas[13][0] * Skin_Sigmas[13][1] * Skin_Sigmas[13][2],
    Skin_Sigmas[14][0] * Skin_Sigmas[14][1] * Skin_Sigmas[14][2],
    Skin_Sigmas[15][0] * Skin_Sigmas[15][1] * Skin_Sigmas[15][2],
};

static constexpr float PrecomputedNonskin_Sigmas_det[16] = {
    Nonskin_Sigmas[0][0] * Nonskin_Sigmas[0][1] * Nonskin_Sigmas[0][2],
    Nonskin_Sigmas[1][0] * Nonskin_Sigmas[1][1] * Nonskin_Sigmas[1][2],
    Nonskin_Sigmas[2][0] * Nonskin_Sigmas[2][1] * Nonskin_Sigmas[2][2],
    Nonskin_Sigmas[3][0] * Nonskin_Sigmas[3][1] * Nonskin_Sigmas[3][2],
    Nonskin_Sigmas[4][0] * Nonskin_Sigmas[4][1] * Nonskin_Sigmas[4][2],
    Nonskin_Sigmas[5][0] * Nonskin_Sigmas[5][1] * Nonskin_Sigmas[5][2],
    Nonskin_Sigmas[6][0] * Nonskin_Sigmas[6][1] * Nonskin_Sigmas[6][2],
    Nonskin_Sigmas[7][0] * Nonskin_Sigmas[7][1] * Nonskin_Sigmas[7][2],
    Nonskin_Sigmas[8][0] * Nonskin_Sigmas[8][1] * Nonskin_Sigmas[8][2],
    Nonskin_Sigmas[9][0] * Nonskin_Sigmas[9][1] * Nonskin_Sigmas[9][2],
    Nonskin_Sigmas[10][0] * Nonskin_Sigmas[10][1] * Nonskin_Sigmas[10][2],
    Nonskin_Sigmas[11][0] * Nonskin_Sigmas[11][1] * Nonskin_Sigmas[11][2],
    Nonskin_Sigmas[12][0] * Nonskin_Sigmas[12][1] * Nonskin_Sigmas[12][2],
    Nonskin_Sigmas[13][0] * Nonskin_Sigmas[13][1] * Nonskin_Sigmas[13][2],
    Nonskin_Sigmas[14][0] * Nonskin_Sigmas[14][1] * Nonskin_Sigmas[14][2],
    Nonskin_Sigmas[15][0] * Nonskin_Sigmas[15][1] * Nonskin_Sigmas[15][2],
};

static const float PrecomputedSkin_GaussCoeff[16] = {
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[0]) / Skin_Ws[0],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[1]) / Skin_Ws[1],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[2]) / Skin_Ws[2],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[3]) / Skin_Ws[3],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[4]) / Skin_Ws[4],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[5]) / Skin_Ws[5],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[6]) / Skin_Ws[6],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[7]) / Skin_Ws[7],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[8]) / Skin_Ws[8],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[9]) / Skin_Ws[9],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[10]) / Skin_Ws[10],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[11]) / Skin_Ws[11],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[12]) / Skin_Ws[12],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[13]) / Skin_Ws[13],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[14]) / Skin_Ws[14],
    std::sqrt(GaussDetCoeff * PrecomputedSkin_Sigmas_det[15]) / Skin_Ws[15]
};

static const float PrecomputedNonskin_GaussCoeff[16] = {
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[0]) / Nonskin_Ws[0],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[1]) / Nonskin_Ws[1],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[2]) / Nonskin_Ws[2],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[3]) / Nonskin_Ws[3],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[4]) / Nonskin_Ws[4],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[5]) / Nonskin_Ws[5],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[6]) / Nonskin_Ws[6],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[7]) / Nonskin_Ws[7],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[8]) / Nonskin_Ws[8],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[9]) / Nonskin_Ws[9],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[10]) / Nonskin_Ws[10],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[11]) / Nonskin_Ws[11],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[12]) / Nonskin_Ws[12],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[13]) / Nonskin_Ws[13],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[14]) / Nonskin_Ws[14],
    std::sqrt(GaussDetCoeff * PrecomputedNonskin_Sigmas_det[15]) / Nonskin_Ws[15]
};
