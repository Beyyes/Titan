import React, { Component, Fragment } from 'react';

export default class AirflowWrapper extends Component {

  render() {
    return (
      <Fragment>
        <iframe src="http://spark-master:30280/" width="100%" height="100%" frameBorder="0" scrolling="no"></iframe>
      </Fragment>
    )
  }
}

