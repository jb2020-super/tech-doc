// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "CHelloCOM.h"
#include "CClassFactory.h"

DEFINE_GUID(CLSID_HelloCOM, 0x24DABF07, 0x7213, 0x4B68, 0xBA, 0x3F, 0xFC, 0xBB, 0xC8, 0x3D, 0xDA, 0x95);

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}


STDAPI
DllGetClassObject(
    __in REFCLSID rClsID,
    __in REFIID riid,
    __deref_out void** pv)
{
    *pv = NULL;
    if (!(riid == IID_IUnknown) && !(riid == IID_IClassFactory)) {
        return E_NOINTERFACE;
    }
    HRESULT hr{ S_OK };
    if (rClsID == CLSID_HelloCOM) {
        CClassFactory<CHelloCOM>* pcf = new CClassFactory<CHelloCOM>();
        if (pcf)
        {
            hr = pcf->QueryInterface(riid, pv);
            pcf->Release();
        }
        else
        {
            hr = E_OUTOFMEMORY;
        }
    }
    else {
        return E_NOTIMPL;
    }

    return hr;
}

STDAPI
DllCanUnloadNow()
{
    return g_lLockCount ? S_FALSE : S_OK;
}