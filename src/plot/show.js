/**
 * Attach lightweight rendering helpers to plot configuration objects.
 * Allows calling config.show(Plot) in Observable without adding direct Plot dependency.
 */

const SUPPORTED_MARKS = new Set([
  'dot',
  'line',
  'area',
  'arrow',
  'text',
  'barX',
  'barY',
  'cell',
  'ruleX',
  'ruleY',
  'rect',
  'rectY'
]);

/**
 * Attach a .show() helper using a custom renderer normalization and render function.
 * @param {Object} config - Visualization configuration object
 * @param {Object} options - { errorMessage, normalizeRenderer, render }
 * @returns {Object} The same config with a non-enumerable show() helper
 */
export function attachCustomShow(
  config,
  {
    errorMessage = 'show() requires a renderer.',
    normalizeRenderer = (renderer) => renderer,
    render
  } = {}
) {
  if (!config || typeof config !== 'object') {
    return config;
  }

  if (typeof render !== 'function') {
    throw new Error('attachCustomShow requires a render(callback) function.');
  }

  if (typeof config.show === 'function') {
    return config;
  }

  Object.defineProperty(config, 'show', {
    enumerable: false,
    value(renderer, overrides = {}) {
      if (!renderer) {
        throw new Error(errorMessage);
      }

      const normalized = normalizeRenderer(renderer);
      return render({
        renderer: normalized,
        config,
        overrides
      });
    }
  });

  return config;
}

/**
 * Attach a .show() helper to the configuration if marks are present.
 * @param {Object} config - Plot configuration object
 * @returns {Object} The same config with a non-enumerable show() helper
 */
export function attachShow(config) {
  if (!config || typeof config !== 'object' || !Array.isArray(config.marks)) {
    return config;
  }

  return attachCustomShow(config, {
    errorMessage: 'show() requires a renderer. Pass the Observable Plot module, e.g. config.show(Plot).',
    normalizeRenderer,
    render({ renderer, config: originalConfig, overrides }) {
      const spec = buildObservablePlotSpec(originalConfig, renderer, overrides);
      return renderer.plot(spec);
    }
  });
}

/**
 * Attach a .show() helper for dendrogram-like visualizations.
 * Allows calling config.show(renderer) where renderer is either a function
 * or an object with `render` or `dendrogram` method.
 * @param {Object} config - Dendrogram visualization configuration
 * @returns {Object} Config with .show() helper
 */
export function attachTreeShow(config) {
  return attachCustomShow(config, {
    errorMessage: 'show() requires a dendrogram renderer. Pass a function or object, e.g. config.show(d3Dendrogram).',
    normalizeRenderer: normalizeTreeRenderer,
    render({ renderer, config: originalSpec, overrides }) {
      const { data: baseData, config: baseConfig = {}, ...restBase } = originalSpec;
      const { data: overrideData, config: overrideConfig, ...restOverrides } = overrides || {};

      const filteredOverrides = { ...restOverrides };
      delete filteredOverrides.type;

      const finalConfig = {
        ...baseConfig,
        ...(typeof filteredOverrides === 'object' ? filteredOverrides : {})
      };

      if (overrideConfig && typeof overrideConfig === 'object') {
        Object.assign(finalConfig, overrideConfig);
      }

      const finalSpec = {
        ...restBase,
        data: overrideData ?? baseData,
        config: finalConfig
      };

      return renderer(finalSpec);
    }
  });
}

function normalizeRenderer(renderer) {
  if (renderer && typeof renderer.plot === 'function') {
    return renderer;
  }

  throw new Error('Unsupported renderer passed to show(). Pass the Observable Plot module (import * as Plot).');
}

function normalizeTreeRenderer(renderer) {
  if (typeof renderer === 'function') {
    return renderer;
  }

  if (renderer && typeof renderer.render === 'function') {
    return renderer.render.bind(renderer);
  }

  if (renderer && typeof renderer.dendrogram === 'function') {
    return renderer.dendrogram.bind(renderer);
  }

  throw new Error('Unsupported dendrogram renderer passed to show(). Provide a function or an object with render() or dendrogram() method.');
}

function buildObservablePlotSpec(config, Plot, overrides) {
  ensureMarksSupported(config);

  const spec = {};
  copyOptionalProps(config, spec, [
    'width',
    'height',
    'margin',
    'marginLeft',
    'marginRight',
    'marginTop',
    'marginBottom',
    'title',
    'subtitle',
    'style'
  ]);

  if (config.axes) {
    if (config.axes.x) spec.x = {...config.axes.x};
    if (config.axes.y) spec.y = {...config.axes.y};
    if (config.axes.fy) spec.fy = {...config.axes.fy};
    if (config.axes.fx) spec.fx = {...config.axes.fx};
  }

  if (config.legend && config.legend.color) {
    spec.color = {
      legend: true,
      ...(config.legend.color.label ? { label: config.legend.color.label } : {}),
      ...(config.legend.color.domain ? { domain: config.legend.color.domain } : {})
    };
  }

  const datasets = config.data || {};

  spec.marks = config.marks.map(mark => convertMarkToPlot(mark, datasets, Plot));

  return {
    ...spec,
    ...overrides,
    marks: overrides.marks ?? spec.marks
  };
}

function copyOptionalProps(source, target, keys) {
  keys.forEach(key => {
    if (source[key] !== undefined) {
      target[key] = source[key];
    }
  });
}

function ensureMarksSupported(config) {
  const unsupported = config.marks
    .map(mark => mark.type)
    .filter(type => type && !SUPPORTED_MARKS.has(type));

  if (unsupported.length > 0) {
    throw new Error(`show() does not support mark types: ${[...new Set(unsupported)].join(', ')}`);
  }
}

function convertMarkToPlot(mark, datasets, Plot) {
  const { type, data: dataKey, tip, ...rest } = mark;
  const options = {...rest};

  // Provide sensible tooltip defaults for marks flagged with tip=true
  if (tip && options.title === undefined) {
    options.title = d => d.label ?? d.index ?? '';
  }

  switch (type) {
    case 'dot':
      return Plot.dot(resolveDataset(type, dataKey, datasets), options);
    case 'line':
      return Plot.line(resolveDataset(type, dataKey, datasets), options);
    case 'area':
      return Plot.area(resolveDataset(type, dataKey, datasets), options);
    case 'arrow':
      return Plot.arrow(resolveDataset(type, dataKey, datasets), options);
    case 'text':
      return Plot.text(resolveDataset(type, dataKey, datasets), options);
    case 'barX':
      return Plot.barX(resolveDataset(type, dataKey, datasets), options);
    case 'barY':
      return Plot.barY(resolveDataset(type, dataKey, datasets), options);
    case 'cell':
      return Plot.cell(resolveDataset(type, dataKey, datasets), options);
    case 'rect':
      return Plot.rect(resolveDataset(type, dataKey, datasets), options);
    case 'rectY':
      return Plot.rectY(resolveDataset(type, dataKey, datasets), options);
    case 'ruleX':
      return renderRule('ruleX', Plot, dataKey, datasets, options);
    case 'ruleY':
      return renderRule('ruleY', Plot, dataKey, datasets, options);
    default:
      throw new Error(`Unsupported mark type '${type}' in show().`);
  }
}

function renderRule(axis, Plot, dataKey, datasets, options) {
  const fn = Plot[axis];
  if (!fn) {
    throw new Error(`Renderer does not support ${axis} marks.`);
  }

  if (dataKey) {
    return fn(resolveDataset(axis, dataKey, datasets), options);
  }

  const valueKey = axis === 'ruleX' ? 'x' : 'y';
  const value = options[valueKey];
  if (value === undefined) {
    throw new Error(`${axis} mark requires either data or a ${valueKey} value.`);
  }

  const arrayValue = Array.isArray(value) ? value : [value];
  const newOptions = {...options};
  delete newOptions[valueKey];
  return fn(arrayValue, newOptions);
}

function resolveDataset(markType, dataKey, datasets) {
  if (!dataKey) {
    throw new Error(`Mark of type '${markType}' is missing a data source.`);
  }

  const data = datasets[dataKey];
  if (!data) {
    throw new Error(`Data set '${dataKey}' not found on config.data for mark type '${markType}'.`);
  }

  return data;
}
