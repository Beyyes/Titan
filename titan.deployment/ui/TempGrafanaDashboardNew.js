import React, { Component, Fragment } from 'react';

export default class GrafanaDashboardNewWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://DEFINE-MASTER-IP:3000/dashboard/new" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

