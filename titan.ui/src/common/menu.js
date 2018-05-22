import { isUrl } from '../utils/utils';

const menuData = [
  {
    name: '控制台',
    icon: 'dashboard',
    path: 'dashboard',
    children: [
      {
        name: '任务管理',
        path: 'airflow',
        // hideInBreadcrumb: true,
        // hideInMenu: true,
      },
      {
        name: '模型服务监控',
        path: 'grafana',
        children: [

          {
            name: '首页',
            path: 'home',
          },
          {
            name: '仪表盘管理',
            path: 'dashboard-manager',
          },
          {
            name: '创建仪表盘',
            path: 'dashboard-new',
          },
          {
            name: '导入仪表盘',
            path: 'dashboard-import',
          },
          {
            name: 'Alert管理',
            path: 'alerts',
          },
          {
            name: '数据源管理',
            path: 'datasources',
          },
        ],
      },
      {
        name: '集群监控',
        path: 'kubernetes',
        authority: 'admin',
      },
    ],
  },
  {
    name: '账户',
    icon: 'user',
    path: 'user',
    authority: 'guest',
    children: [
      {
        name: '登录',
        path: 'login',
      },
      {
        name: '注册',
        path: 'register',
      },
      {
        name: '注册结果',
        path: 'register-result',
      },
    ],
  },
      /*
  {
    name: '异常页',
    icon: 'warning',
    path: 'exception',
    children: [
      {
        name: '403',
        path: '403',
      },
      {
        name: '404',
        path: '404',
      },
      {
        name: '500',
        path: '500',
      },
      {
        name: '触发异常',
        path: 'trigger',
        hideInMenu: true,
      },
    ],
  },
  */
];

function formatter(data, parentPath = '/', parentAuthority) {
  return data.map(item => {
    let { path } = item;
    if (!isUrl(path)) {
      path = parentPath + item.path;
    }
    const result = {
      ...item,
      path,
      authority: item.authority || parentAuthority,
    };
    if (item.children) {
      result.children = formatter(item.children, `${parentPath}${item.path}/`, item.authority);
    }
    return result;
  });
}

export const getMenuData = () => formatter(menuData);
