import React, { Component, Fragment } from 'react';

export default class GrafanaDashboardNewFolderWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://localhost:3333/dashboards/folder/new" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

