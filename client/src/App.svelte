<script>
  import { Router, Route } from 'svelte-navigator';
  import Home from './routes/Home.svelte';
  import Results from './routes/Results.svelte';
</script>

<Router>
  <div class="container">
    <header class="text-center mb-4">
      <h1>Neuralrad Mammo AI</h1>
    </header>
    
    <main>
      <Route path="/" component={Home} />
      <Route path="/results" component={Results} />
    </main>
    
    <footer class="text-center mt-4">
      <p>Mammography Analysis Tool © 2025</p>
    </footer>
  </div>
</Router>
