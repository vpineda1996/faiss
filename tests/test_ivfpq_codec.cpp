/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <omp.h>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/utils/distances.h>

namespace {

// dimension of the vectors to index
int d = 64;

// size of the database we plan to index
size_t nb = 8000;

double eval_codec_error(long ncentroids, long m, const std::vector<float>& v) {
    faiss::IndexFlatL2 coarse_quantizer(d);
    faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, m, 8);
    index.pq.cp.niter = 10; // speed up train
    index.train(nb, v.data());

    // encode and decode to compute reconstruction error

    std::vector<faiss::idx_t> keys(nb);
    std::vector<uint8_t> codes(nb * m);
    index.encode_multiple(nb, keys.data(), v.data(), codes.data(), true);

    std::vector<float> v2(nb * d);
    index.decode_multiple(nb, keys.data(), codes.data(), v2.data());

    return faiss::fvec_L2sqr(v.data(), v2.data(), nb * d);
}

} // namespace

bool runs_on_sandcastle() {
    // see discussion here https://fburl.com/qc5kpdo2
    const char* sandcastle = getenv("SANDCASTLE");
    if (sandcastle && !strcmp(sandcastle, "1")) {
        return true;
    }
    const char* tw_job_user = getenv("TW_JOB_USER");
    if (tw_job_user && !strcmp(tw_job_user, "sandcastle")) {
        return true;
    }

    return false;
}

TEST(IndexPQ, codec) {
    std::vector<float> database(nb * d);
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }

    // limit number of threads when running on heavily parallelized test
    // environment
    if (runs_on_sandcastle()) {
        omp_set_num_threads(2);
    }

    // Create and train the IndexPQ
    int m = 8; // number of subquantizers
    int nbits = 8; // bits per subquantizer
    faiss::IndexPQ index(d, m, nbits);
    index.train(1500, database.data());

    // Encode and decode to compute reconstruction error
    std::vector<uint8_t> codes(nb * m);

    index.add(1, database.data());
    index.add(1, database.data() + d);
    index.add(1, database.data() + 2 * d);
    index.add(1, database.data() + 3 * d);

    for (int i = 0; i < 49; i++) {
        std::vector<float> v(database.data() + 4 * d, database.data() + 5 * d);
        v[0] = distrib(rng) + 100;
        index.add(1, v.data());
    }

    // add a datapoint that skews the average higher
    std::vector<float> v(database.data() + 4 * d, database.data() + 5 * d);
    v[0] = distrib(rng) + 200;
    index.add(1, v.data());

    // modify 0th coord of the 4th vector to fall within the radius
    database[4 * d] = 100;

    size_t k = 4;
    size_t nprobe = 5;
    std::vector<faiss::idx_t> neighbors(nprobe * k * d);
    std::vector<float> distances(nprobe * k);
    std::vector<size_t> freq(nprobe * k);
    
    index.search_neighbourhood(nprobe, database.data(), k, distances.data(), neighbors.data(), freq.data());

    for (int i = 0; i < nprobe; i++) {
        for (int j = 0; j < k; j++) {
            printf("[i=%d][k=%d] Distance: %f, Neighbor: %ld, Frequency: %zu\n", i, j, distances[i * k + j], neighbors[i * k + j], freq[i * k + j]);
        }
    }

    // first distance should be zero for the first k vectors
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(distances[i * k], 0);
        EXPECT_EQ(neighbors[i * k], i);
        EXPECT_EQ(freq[i * k], 1);
    }

    // within radius
    EXPECT_EQ(neighbors[k * k], 4);
    EXPECT_EQ(distances[k * k], 0);

    // k+1... vector should have a non-zero distance to its first neighbor
    for (int i = k+1; i < nprobe; i++) {
        EXPECT_GT(distances[i * k], 0);
    }

    // outside radius by avg
    database[4 * d] = 120;
    index.search_neighbourhood(nprobe, database.data(), k, distances.data(), neighbors.data(), freq.data());
    EXPECT_EQ(neighbors[k * k], 4);
    // l2 should be greater than 100 
    EXPECT_GT(distances[k * k], 100*100);
}

TEST(IVFPQ, codec) {
    std::vector<float> database(nb * d);
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }

    // limit number of threads when running on heavily parallelized test
    // environment
    if (runs_on_sandcastle()) {
        omp_set_num_threads(2);
    }

    double err0 = eval_codec_error(16, 8, database);

    // should be more accurate as there are more coarse centroids
    double err1 = eval_codec_error(128, 8, database);
    EXPECT_GT(err0, err1);

    // should be more accurate as there are more PQ codes
    double err2 = eval_codec_error(16, 16, database);
    EXPECT_GT(err0, err2);
}
