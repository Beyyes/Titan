import React, { Component, Fragment } from 'react';

export default class GrafanaAlertListWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://spark-slave1:3000/alerting" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

