1. install anaconda
2. set proxy in the config file c:/user/[username]/.condarc, with:
    ```
    proxy_servers:
        http: http://[username]:[password]@192.168.0.0:[port]
        https: https://[username]:[password]@192.168.0.0:[port]
    ```
3. `conda install -c pytorch pytorch torchvision`