import React, { Component, Fragment } from 'react';

export default class GrafanaDashboardImportWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://spark-slave1:3000/dashboard/import" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

