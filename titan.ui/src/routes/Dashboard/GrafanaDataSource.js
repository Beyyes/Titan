import React, { Component, Fragment } from 'react';

export default class GrafanaDataSourceWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://localhost:3333/datasources" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

