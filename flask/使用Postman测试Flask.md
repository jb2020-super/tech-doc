1. 下载并安装 `Postman App`, [点这里下载](https://www.postman.com/downloads/);
2. 下载 `InterceptorBridge`, [点这里下载](https://go.pstmn.io/interceptor-bridge-windows)；解压后运行 `install_host.bat`;
3. 在谷歌 Web App 商店里搜索并安装 `Postman Interceptor`；在chrome右上角的插件按钮里点进去，打开 `Capture Requests`；
4. 运行 `python test_flask.py`；
5. 打开 `Postman App`；登录；进入workspace；打开一个新的标签；
6. 输入地址 http://127.0.0.1:5000/；切换到Post；在Body标签里选择raw，输入JSON数据：

    ```json
    {    
        "name": "miaomi",
        "year": 1998
    }
    ```