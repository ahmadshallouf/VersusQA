import { Component} from '@angular/core';
import {AppStateService} from "../../StateManagement/app-state.service";
import {map} from "rxjs";

@Component({
  selector: 'app-user-interface',
  templateUrl: './user-interface.component.html',
  styleUrls: ['./user-interface.component.css']
})

export class UserInterfaceComponent {

  viewState = this.state.state$.pipe(
    map(state => state.viewState)
  );

  showLoading = false; // boolean that checks if the loading screen should be shown
  constructor(public state: AppStateService) {
    this.state.state$.subscribe(state => {
      this.showLoading = state.viewState.processing;
    });
  }

}
