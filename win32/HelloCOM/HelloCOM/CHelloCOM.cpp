#include "pch.h"
#include "CHelloCOM.h"

HRESULT CHelloCOM::CreateInstance(REFIID iid, void** ppv)
{
    CHelloCOM* pobj = new CHelloCOM();
    if (!pobj) {
        return E_OUTOFMEMORY;
    }

    return pobj->QueryInterface(iid, ppv);
}

HRESULT CHelloCOM::QueryInterface(REFIID riid, void** ppv)
{
    if (!ppv)
    {
        return E_POINTER;
    }
    if (riid == IID_IUnknown)
    {
        *ppv = static_cast<IUnknown*>(this);
    }
    else if (riid == __uuidof(IHelloCOM))
    {
        *ppv = static_cast<IHelloCOM*>(this);
    }
    else
    {
        *ppv = NULL;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

ULONG CHelloCOM::AddRef(void)
{
    return InterlockedIncrement(&m_lRefCount);
}

ULONG CHelloCOM::Release(void)
{
    ULONG lRefCount = InterlockedDecrement(&m_lRefCount);
    if (lRefCount == 0)
    {
        delete this;
    }
    return lRefCount;
}

HRESULT CHelloCOM::get_Color(PrintColor* pColor)
{
    return E_NOTIMPL;
}

HRESULT CHelloCOM::get_Position(Point2D* pos)
{
    return E_NOTIMPL;
}

HRESULT CHelloCOM::put_Position(Point2D pos)
{
    return E_NOTIMPL;
}

HRESULT CHelloCOM::Print(BSTR msg)
{
    MessageBoxW(NULL, msg, L"Info", MB_OK);
    return S_OK;
}
