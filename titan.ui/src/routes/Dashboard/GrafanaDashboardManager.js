import React, { Component, Fragment } from 'react';

export default class GrafanaDashboardManageWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://spark-master:30061/dashboards" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

