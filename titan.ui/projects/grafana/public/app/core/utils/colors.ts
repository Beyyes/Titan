import _ from 'lodash';
import tinycolor from 'tinycolor2';

export const PALETTE_ROWS = 3;
export const PALETTE_COLUMNS = 6;
export const DEFAULT_ANNOTATION_COLOR = 'rgba(0, 211, 255, 1)';
export const OK_COLOR = 'rgba(11, 237, 50, 1)';
export const ALERTING_COLOR = 'rgba(237, 46, 24, 1)';
export const NO_DATA_COLOR = 'rgba(150, 150, 150, 1)';
export const REGION_FILL_ALPHA = 0.09;

let colors = [
  '#4472c4',
  '#a5a5a5',
  '#5b9bd5',
  '#264478',
  '#636363',
  '#255e91',
  '#70ad47',
  '#ed7d31',
  '#ffc000',
  '#9e480e',
  '#997300',
  '#43682b',
  '#70ad47',
  '#5b9bd5',
  '#ffc000',
  '#43682b',
  '#255e91',
  '#997300',
];

export function sortColorsByHue(hexColors) {
  let hslColors = _.map(hexColors, hexToHsl);

  let sortedHSLColors = _.sortBy(hslColors, ['h']);
  sortedHSLColors = _.chunk(sortedHSLColors, PALETTE_ROWS);
  sortedHSLColors = _.map(sortedHSLColors, chunk => {
    return _.sortBy(chunk, 'l');
  });
  sortedHSLColors = _.flattenDeep(_.zip(...sortedHSLColors));

  return _.map(sortedHSLColors, hslToHex);
}

export function hexToHsl(color) {
  return tinycolor(color).toHsl();
}

export function hslToHex(color) {
  return tinycolor(color).toHexString();
}

export let sortedColors = sortColorsByHue(colors);
export default colors;
