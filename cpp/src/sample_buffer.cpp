#include "sample_buffer.h"
#include "data_representation.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

SampleBuffer::SampleBuffer(const unsigned long capacity) : capacity(capacity), count(0)
{
    this->obs = new float[GetObsSize(this->capacity)];
    this->act = new int8_t[this->capacity];
    this->val = new float[this->capacity];
}

SampleBuffer::~SampleBuffer() {
    delete[] this->obs;
    delete[] this->act;
    delete[] this->val;
}

ulong SampleBuffer::addSamples(const SampleBuffer& otherBuffer, const ulong offset, const ulong n)
{
    ulong numSamples = std::min(n, this->capacity - this->count);

    std::copy_n(otherBuffer.obs + GetObsSize(offset), GetObsSize(numSamples), this->obs + GetObsSize(this->count));
    std::copy_n(otherBuffer.act + offset, numSamples, this->act + this->count);
    std::copy_n(otherBuffer.val + offset, numSamples, this->val + this->count);

    this->count += numSamples;

    return numSamples;
}

bool SampleBuffer::addSample(const float* planes, const bboard::Move move)
{
    if(count >= capacity)
        return false;

    std::copy_n(planes, GetObsSize(1), this->obs + GetObsSize(this->count));
    act[count] = int8_t(move);
    count += 1;

    return true;
}

void SampleBuffer::setValues(const float value)
{
    std::fill_n(val, count, value);
}

void SampleBuffer::clear() {
    this->count = 0;
}

const float* SampleBuffer::getObs() const {
    return this->obs;
}

const int8_t* SampleBuffer::getAct() const {
    return this->act;
}

const float* SampleBuffer::getVal() const {
    return this->val;
}

ulong SampleBuffer::getCount() const {
    return this->count;
}

ulong SampleBuffer::getCapacity() const {
    return this->capacity;
}

ulong SampleBuffer::getTotalObsValCount() const {
    return GetObsSize(this->getCount());
}
