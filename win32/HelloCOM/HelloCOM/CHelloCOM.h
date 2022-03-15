#pragma once
#include "IHelloCOM.h"

class CHelloCOM :
    public IHelloCOM
{
public:
    static HRESULT CreateInstance(REFIID iid, void** ppv);

    // Inherited via IHelloCOM
    STDMETHODIMP QueryInterface(REFIID riid, void** ppvObject) override;

    STDMETHODIMP_(ULONG) AddRef(void) override;

    STDMETHODIMP_(ULONG) Release(void) override;

    STDMETHODIMP get_Color(PrintColor* pColor) override;

    STDMETHODIMP get_Position(Point2D* pos) override;

    STDMETHODIMP put_Position(Point2D pos) override;

    STDMETHODIMP Print(BSTR msg) override;

private:
    long m_lRefCount{ 0 };
};

