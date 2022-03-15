#pragma once
#include <Unknwnbase.h>

static volatile long g_lLockCount = 0;

template<class T>
class CClassFactory :
    public IClassFactory
{
public:
    static BOOL IsLocked(void) {
        return (g_lLockCount == 0) ? FALSE : TRUE;
    }

    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) {
        if (!ppv)
        {
            return E_POINTER;
        }
        if (iid == IID_IUnknown)
        {
            *ppv = static_cast<IUnknown*>(static_cast<IClassFactory*>(this));
        }
        else if (iid == __uuidof(IClassFactory))
        {
            *ppv = static_cast<IClassFactory*>(this);
        }
        else
        {
            *ppv = NULL;
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }
    STDMETHODIMP_(ULONG) AddRef(void) {
        return InterlockedIncrement(&m_lRefCount);
    }
    STDMETHODIMP_(ULONG) Release(void) {
        ULONG lRefCount = InterlockedDecrement(&m_lRefCount);
        if (lRefCount == 0)
        {
            delete this;
        }
        return lRefCount;
    }

    // IClassFactory
    STDMETHODIMP CreateInstance(LPUNKNOWN punkOuter, REFIID iid, void** ppv) {
        if (ppv == NULL)
        {
            return E_POINTER;
        }

        *ppv = NULL;

        if (punkOuter != NULL)
        {
            return CLASS_E_NOAGGREGATION;
        }

        return T::CreateInstance(iid, ppv);
    }
    STDMETHODIMP LockServer(BOOL fLock) {
        if (fLock == FALSE)
        {
            InterlockedDecrement(&g_lLockCount);
        }
        else
        {
            InterlockedIncrement(&g_lLockCount);
        }
        return S_OK;
    }
private:


    long m_lRefCount{1};
};

