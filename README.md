# Random-forest-nav
Playing with random forest regression and checking out mean average errors (mae) with different node selections

###Output
mae no node specified: 207190.6873773146;
mae with 5 leaf nodes: 375038.0474733729;
mae with 500 leaf nodes: 208580.49362309175;
mae with 5000 leaf nodes: 207143.62285376125;
mae with 50000 leaf nodes: 207143.62285376125;

Random forest regression without a node selected
    will give a prediction (mae 207190, no nodes specified)
       as close to the best prediction (mae 207143)
            with 500 leaf nodes selected
