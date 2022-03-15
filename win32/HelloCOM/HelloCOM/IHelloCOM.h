

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.01.0622 */
/* at Tue Jan 19 11:14:07 2038
 */
/* Compiler settings for IHelloCOM.idl:
    Oicf, W1, Zp8, env=Win64 (32b run), target_arch=AMD64 8.01.0622 
    protocol : all , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */



/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif /* __RPCNDR_H_VERSION__ */

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __IHelloCOM_h__
#define __IHelloCOM_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __HelloCOM_FWD_DEFINED__
#define __HelloCOM_FWD_DEFINED__

#ifdef __cplusplus
typedef class HelloCOM HelloCOM;
#else
typedef struct HelloCOM HelloCOM;
#endif /* __cplusplus */

#endif 	/* __HelloCOM_FWD_DEFINED__ */


#ifndef __IHelloCOM_FWD_DEFINED__
#define __IHelloCOM_FWD_DEFINED__
typedef interface IHelloCOM IHelloCOM;

#endif 	/* __IHelloCOM_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 



#ifndef __HelloCOMLib_LIBRARY_DEFINED__
#define __HelloCOMLib_LIBRARY_DEFINED__

/* library HelloCOMLib */
/* [version][uuid] */ 

typedef /* [public][public] */ 
enum __MIDL___MIDL_itf_IHelloCOM_0000_0000_0001
    {
        PRINT_COLOR_RED	= 0,
        PRINT_COLOR_GREEN	= ( PRINT_COLOR_RED + 1 ) ,
        PRINT_COLOR_BLUE	= ( PRINT_COLOR_GREEN + 1 ) ,
        PRINT_COLOR_BLACK	= ( PRINT_COLOR_BLUE + 1 ) 
    } 	PrintColor;

typedef /* [public][public][public] */ struct __MIDL___MIDL_itf_IHelloCOM_0000_0000_0002
    {
    UINT32 width;
    UINT32 height;
    } 	Point2D;


EXTERN_C const IID LIBID_HelloCOMLib;

EXTERN_C const CLSID CLSID_HelloCOM;

#ifdef __cplusplus

class DECLSPEC_UUID("24DABF07-7213-4B68-BA3F-FCBBC83DDA95")
HelloCOM;
#endif
#endif /* __HelloCOMLib_LIBRARY_DEFINED__ */

#ifndef __IHelloCOM_INTERFACE_DEFINED__
#define __IHelloCOM_INTERFACE_DEFINED__

/* interface IHelloCOM */
/* [uuid][object] */ 


EXTERN_C const IID IID_IHelloCOM;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("0F87688B-EF6B-4780-824E-7A0A2E8647E0")
    IHelloCOM : public IUnknown
    {
    public:
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Color( 
            /* [retval][out] */ PrintColor *pColor) = 0;
        
        virtual /* [propget] */ HRESULT STDMETHODCALLTYPE get_Position( 
            /* [retval][out] */ Point2D *pos) = 0;
        
        virtual /* [propput] */ HRESULT STDMETHODCALLTYPE put_Position( 
            /* [in] */ Point2D pos) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE Print( 
            /* [in] */ BSTR msg) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct IHelloCOMVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IHelloCOM * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IHelloCOM * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IHelloCOM * This);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Color )( 
            IHelloCOM * This,
            /* [retval][out] */ PrintColor *pColor);
        
        /* [propget] */ HRESULT ( STDMETHODCALLTYPE *get_Position )( 
            IHelloCOM * This,
            /* [retval][out] */ Point2D *pos);
        
        /* [propput] */ HRESULT ( STDMETHODCALLTYPE *put_Position )( 
            IHelloCOM * This,
            /* [in] */ Point2D pos);
        
        HRESULT ( STDMETHODCALLTYPE *Print )( 
            IHelloCOM * This,
            /* [in] */ BSTR msg);
        
        END_INTERFACE
    } IHelloCOMVtbl;

    interface IHelloCOM
    {
        CONST_VTBL struct IHelloCOMVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IHelloCOM_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define IHelloCOM_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define IHelloCOM_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define IHelloCOM_get_Color(This,pColor)	\
    ( (This)->lpVtbl -> get_Color(This,pColor) ) 

#define IHelloCOM_get_Position(This,pos)	\
    ( (This)->lpVtbl -> get_Position(This,pos) ) 

#define IHelloCOM_put_Position(This,pos)	\
    ( (This)->lpVtbl -> put_Position(This,pos) ) 

#define IHelloCOM_Print(This,msg)	\
    ( (This)->lpVtbl -> Print(This,msg) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __IHelloCOM_INTERFACE_DEFINED__ */


/* Additional Prototypes for ALL interfaces */

unsigned long             __RPC_USER  BSTR_UserSize(     unsigned long *, unsigned long            , BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserMarshal(  unsigned long *, unsigned char *, BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserUnmarshal(unsigned long *, unsigned char *, BSTR * ); 
void                      __RPC_USER  BSTR_UserFree(     unsigned long *, BSTR * ); 

unsigned long             __RPC_USER  BSTR_UserSize64(     unsigned long *, unsigned long            , BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserMarshal64(  unsigned long *, unsigned char *, BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserUnmarshal64(unsigned long *, unsigned char *, BSTR * ); 
void                      __RPC_USER  BSTR_UserFree64(     unsigned long *, BSTR * ); 

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


