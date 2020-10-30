#ifndef CLONABLE_H
#define CLONABLE_H

#include <cstdlib>
#include <memory>

/**
 * @brief A clonable object.
 */
template <typename T>
class Clonable
{
public:
    /**
     * @brief get_obj_ptr Get the pointer to the instance of the underlying object.
     * @return The object pointer
     */
    virtual T* get_obj_ptr() = 0;

    /**
     * @brief clone Create a clonable clone of the underlying object.
     * @return A clonable clone of the object.
     */
    virtual std::unique_ptr<Clonable<T>> clone() = 0;
};

/**
 * @brief A wrapper for an object which can be cloned by copying it.
 */
template <typename BaseType, typename ObjectType=BaseType>
class CopyClonable : public Clonable<BaseType>, public ObjectType
{
public:
    CopyClonable(ObjectType obj) {
        static_assert(std::is_base_of<BaseType, ObjectType>::value, "ObjectType not derived from BaseType");
        *get_obj_ptr() = obj;
    }

    CopyClonable(ObjectType* obj) {
        *get_obj_ptr() = *obj;
    }

    ObjectType* get_obj_ptr() override {
        return this;
    }

    std::unique_ptr<Clonable<BaseType>> clone() override {
        // copying is done automatically in the constructor
        return std::make_unique<CopyClonable<BaseType, ObjectType>>(this);;
    }
};

#endif // CLONABLE_H
