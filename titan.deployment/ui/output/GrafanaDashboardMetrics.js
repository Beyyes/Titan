/*eslint no-script-url: "error"*/
import React, { Component } from 'react';
import $ from 'jquery';
import _ from 'underscore';
import { Table, Icon, Button, InputNumber } from 'antd';

function fetchData() {
  let res = null;
  $.ajax('http://spark-slave1:8088/ws/v1/cluster/apps', {
    accepts: { json: 'application/json' },
    dataType: 'json',
    async: false,
    success: function(data) {
      const { apps } = data;
      res = apps;
    },
    error: function(xhr, status, err) {
      throw new Error(err);
    },
  });
  return res;
}

function getData() {
  let apps = fetchData().app;
  const data = _.map(apps, app => ({
    id: app.id,
    user: app.user,
    name: app.name,
  }));
  return data;
}

export default class MetricsMonitorPage extends Component {
  state = {
    isHome: true,
    url: '',
  };
  columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      render: text => (
        <a href="javascript:void(0);" onClick={e => this.handleClick(e, text)}>
          {text}
        </a>
      ),
    },
    {
      title: 'User',
      dataIndex: 'user',
      key: 'user',
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
  ];

  handleClick = (e, appId) => {
    e.preventDefault();
    this.url = `http://spark-slave1:3333/dashboard/script/spark.js?app=${appId}&maxExecutorId=2&orgId=1&from=now-6h&to=now`;
    if (this.setState) {
      this.setState({ isHome: false, url: this.url });
    }
  };

  handleGoBack = e => {
    e.preventDefault();
    this.setState({ isHome: true });
  };

  onChange = (value) => {
    const replacer = (match, p1, p2, p3) => `${p1}${value}${p3}`;
    this.url = this.url.replace(/^(.*maxExecutorId=)(\d*)(&.*)$/, replacer);
    this.setState({ isHome: false, url: this.url });
  };

  render() {
    return this.state.isHome ? (
      <Table columns={this.columns} dataSource={getData()} />
    ) : (
      <div>
        <Button type="primary" onClick={this.handleGoBack}>
          <Icon type="left" />Backward
        </Button>
        <div>
          <label>ExecutorNumber: </label>
          <InputNumber min={1} max={100} defaultValue={2} onChange={this.onChange} />
        </div>
        <iframe
          title="metrics-monitor"
          src={this.state.url}
          width="100%"
          height="800px"
          frameBorder="0"
        />
      </div>
    );
  }
}
